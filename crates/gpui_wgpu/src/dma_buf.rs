//! Video frames in DMA-BUFs, imported into wgpu's Vulkan backend as the luma and chroma
//! textures of the surface shader.

use std::collections::BTreeSet;
use std::sync::Arc;

use anyhow::{Context as _, Result, anyhow, ensure};
use ash::{ext, khr, vk};
use gpui::{DmaBufFrame, DmaBufImport, DmaBufPlane};

/// How many drawn frames an import outlives its last draw, so that a frame the GPU may
/// still sample stays alive and unchanged.
const KEPT_FRAMES: u64 = 3;

/// The imports of the frames surfaces drew lately.
#[derive(Default)]
pub(crate) struct ImportedFrames {
    frames: Vec<ImportedFrame>,
    drawn: u64,
    /// The last import's failure, logged once rather than at every draw.
    last_error: Option<String>,
}

struct ImportedFrame {
    frame: Arc<DmaBufFrame>,
    luma: wgpu::TextureView,
    chroma: wgpu::TextureView,
    last_drawn: u64,
}

impl ImportedFrames {
    /// Starts a drawn frame, and drops the imports that stopped drawing.
    pub(crate) fn begin_frame(&mut self) {
        self.drawn += 1;
        let drawn = self.drawn;
        self.frames
            .retain(|frame| drawn - frame.last_drawn <= KEPT_FRAMES);
    }

    /// Returns the luma and chroma textures of `frame`, importing it on its first draw, or
    /// `None` when the import fails.
    pub(crate) fn textures(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: &Arc<DmaBufFrame>,
    ) -> Option<(wgpu::TextureView, wgpu::TextureView)> {
        match self.import(device, queue, frame) {
            Ok(textures) => {
                self.last_error = None;
                Some(textures)
            }
            Err(error) => {
                let error = format!("{error:#}");
                if self.last_error.as_ref() != Some(&error) {
                    log::warn!("a surface draws nothing: {error}");
                    self.last_error = Some(error);
                }
                None
            }
        }
    }

    fn import(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: &Arc<DmaBufFrame>,
    ) -> Result<(wgpu::TextureView, wgpu::TextureView)> {
        if let Some(imported) = self
            .frames
            .iter_mut()
            .find(|imported| Arc::ptr_eq(&imported.frame, frame))
        {
            imported.last_drawn = self.drawn;
            return Ok((imported.luma.clone(), imported.chroma.clone()));
        }
        ensure!(
            device
                .features()
                .contains(wgpu::Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF),
            "the GPU imports no DMA-BUFs"
        );
        ensure!(
            frame.coding.bit_depth == 8,
            "surfaces draw 8-bit frames, not {}-bit ones",
            frame.coding.bit_depth
        );
        let limit = device.limits().max_texture_dimension_2d;
        ensure!(
            (1..=limit).contains(&frame.width) && (1..=limit).contains(&frame.height),
            "a {} by {} frame does not fit the GPU's textures of up to {limit} texels",
            frame.width,
            frame.height
        );
        // The import trusts that the driver samples R8 and RG8 images at the frame's
        // modifier. Drivers list the modifiers they take, and lavapipe lists none, though it
        // imports and draws them, so the check belongs with the producer's choice of layout.
        let (luma, luma_image) = import_plane(
            device,
            &frame.luma,
            wgpu::TextureFormat::R8Unorm,
            frame.width,
            frame.height,
        )?;
        let (chroma, chroma_image) = import_plane(
            device,
            &frame.chroma,
            wgpu::TextureFormat::Rg8Unorm,
            frame.width.div_ceil(2),
            frame.height.div_ceil(2),
        )?;
        acquire(device, queue, &[luma_image, chroma_image])?;
        self.frames.push(ImportedFrame {
            frame: frame.clone(),
            luma: luma.clone(),
            chroma: chroma.clone(),
            last_drawn: self.drawn,
        });
        Ok((luma, chroma))
    }
}

/// Imports `plane` as a texture of `format`, `width` by `height` texels, and returns its view
/// with its Vulkan image.
fn import_plane(
    device: &wgpu::Device,
    plane: &DmaBufPlane,
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
) -> Result<(wgpu::TextureView, ash::vk::Image)> {
    let size = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    let descriptor = wgpu::TextureDescriptor {
        label: Some("dma_buf_plane"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    };
    let hal_descriptor = wgpu::hal::TextureDescriptor {
        label: descriptor.label,
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUses::RESOURCE,
        memory_flags: wgpu::hal::MemoryFlags::empty(),
        view_formats: Vec::new(),
    };
    // Vulkan takes the descriptor it imports, so the frame keeps its own.
    let fd = plane.fd.try_clone().context("duplicating a DMA-BUF")?;
    // SAFETY: the guard only imports into the device it came from, and drops before the
    // texture is wrapped.
    let hal = unsafe { device.as_hal::<wgpu::hal::api::Vulkan>() }
        .context("the device is not a Vulkan device")?;
    // SAFETY: the device has the DMA-BUF feature, `fd` is a DMA-BUF, and the frame's
    // producer vouches for the modifier, stride and offset of the plane.
    let texture = unsafe {
        hal.texture_from_dmabuf_fd(
            fd,
            &hal_descriptor,
            plane.modifier,
            plane.stride.into(),
            plane.offset.into(),
        )
    }
    .map_err(|error| anyhow!("importing a {format:?} plane of a DMA-BUF: {error:?}"))?;
    drop(hal);
    // SAFETY: the texture is alive until it is wrapped below.
    let image = unsafe { texture.raw_handle() };
    // SAFETY: the texture comes from this device with `hal_descriptor`, which matches
    // `descriptor`. `acquire` moves the image into the shader-read layout claimed here
    // before any draw samples it, and the producer wrote its texels before handing the
    // frame over.
    let texture = unsafe {
        device.create_texture_from_hal::<wgpu::hal::api::Vulkan>(
            texture,
            &descriptor,
            wgpu::TextureUses::RESOURCE,
            true,
        )
    };
    Ok((
        texture.create_view(&wgpu::TextureViewDescriptor::default()),
        image,
    ))
}

/// Takes `images` over from the producer of their DMA-BUFs, in the layout shaders read,
/// on the queue's next submission.
fn acquire(device: &wgpu::Device, queue: &wgpu::Queue, images: &[ash::vk::Image]) -> Result<()> {
    // SAFETY: the guard only lends the device's handles, and drops before the encoder.
    let hal = unsafe { device.as_hal::<wgpu::hal::api::Vulkan>() }
        .context("the device is not a Vulkan device")?;
    let raw_device = hal.raw_device().clone();
    let queue_family = hal.queue_family_index();
    drop(hal);
    // wgpu enables no VK_EXT_queue_family_foreign, so the external family stands for the
    // decoder, and the general layout keeps the texels it wrote.
    let barriers: Vec<_> = images
        .iter()
        .map(|image| {
            vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_EXTERNAL)
                .dst_queue_family_index(queue_family)
                .image(*image)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1),
                )
        })
        .collect();
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("dma_buf_acquire"),
    });
    // SAFETY: the barrier only touches the imported images, which wgpu owns and no other
    // command uses before this submission.
    let recorded = unsafe {
        encoder.as_hal_mut::<wgpu::hal::api::Vulkan, _, _>(|encoder| {
            let encoder = encoder?;
            raw_device.cmd_pipeline_barrier(
                encoder.raw_handle(),
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barriers,
            );
            Some(())
        })
    };
    recorded.context("the command encoder is not recording")?;
    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

pub(crate) fn query_import(adapter: &wgpu::Adapter, device: &wgpu::Device) -> Option<DmaBufImport> {
    if !device
        .features()
        .contains(wgpu::Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF)
    {
        return None;
    }
    let import = query_properties(adapter).unwrap_or_else(|error| {
        log::warn!("the GPU's DMA-BUF import is unknown: {error:#}");
        DmaBufImport::default()
    });
    Some(import)
}

fn query_properties(adapter: &wgpu::Adapter) -> Result<DmaBufImport> {
    // SAFETY: the guard only lends the adapter's handles to the queries below.
    let adapter = unsafe { adapter.as_hal::<wgpu::hal::api::Vulkan>() }
        .context("the adapter is not a Vulkan adapter")?;
    let shared = adapter.shared_instance();
    ensure!(
        shared
            .extensions()
            .contains(&khr::get_physical_device_properties2::NAME),
        "the Vulkan instance lacks VK_KHR_get_physical_device_properties2"
    );
    let instance =
        khr::get_physical_device_properties2::Instance::new(shared.entry(), shared.raw_instance());
    let physical_device = adapter.raw_physical_device();
    let render_node = if adapter
        .physical_device_capabilities()
        .supports_extension(ext::physical_device_drm::NAME)
    {
        query_render_node(&instance, physical_device)
    } else {
        None
    };
    let luma = query_modifiers(&instance, physical_device, vk::Format::R8_UNORM);
    let chroma = query_modifiers(&instance, physical_device, vk::Format::R8G8_UNORM);
    let import = DmaBufImport {
        render_node,
        modifiers: luma.intersection(&chroma).copied().collect(),
    };
    Ok(import)
}

fn query_render_node(
    instance: &khr::get_physical_device_properties2::Instance,
    physical_device: vk::PhysicalDevice,
) -> Option<(u32, u32)> {
    let mut drm = vk::PhysicalDeviceDrmPropertiesEXT::default();
    // SAFETY: the physical device has VK_EXT_physical_device_drm, and its properties extend
    // the query.
    unsafe {
        instance.get_physical_device_properties2(
            physical_device,
            &mut vk::PhysicalDeviceProperties2::default().push_next(&mut drm),
        );
    }
    if drm.has_render != vk::TRUE {
        return None;
    }
    let render_node = u32::try_from(drm.render_major)
        .ok()
        .zip(u32::try_from(drm.render_minor).ok());
    if render_node.is_none() {
        log::warn!(
            "the GPU's render node {}:{} is out of range",
            drm.render_major,
            drm.render_minor
        );
    }
    render_node
}

fn query_modifiers(
    instance: &khr::get_physical_device_properties2::Instance,
    physical_device: vk::PhysicalDevice,
    format: vk::Format,
) -> BTreeSet<u64> {
    let mut list = vk::DrmFormatModifierPropertiesListEXT::default();
    // SAFETY: DMA-BUF import implies VK_EXT_image_drm_format_modifier, and its list extends the
    // query. A list without an array takes only the count.
    unsafe {
        instance.get_physical_device_format_properties2(
            physical_device,
            format,
            &mut vk::FormatProperties2::default().push_next(&mut list),
        );
    }
    let mut properties = vec![
        vk::DrmFormatModifierPropertiesEXT::default();
        list.drm_format_modifier_count as usize
    ];
    let mut list = vk::DrmFormatModifierPropertiesListEXT::default()
        .drm_format_modifier_properties(&mut properties);
    // SAFETY: as above, and the list's count is the length of its array.
    unsafe {
        instance.get_physical_device_format_properties2(
            physical_device,
            format,
            &mut vk::FormatProperties2::default().push_next(&mut list),
        );
    }
    let written = list.drm_format_modifier_count as usize;
    properties.truncate(written);
    // The surface shader samples both planes with linear filtering.
    let features =
        vk::FormatFeatureFlags::SAMPLED_IMAGE | vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR;
    properties
        .into_iter()
        .filter(|modifier| {
            modifier.drm_format_modifier_plane_count == 1
                && modifier
                    .drm_format_modifier_tiling_features
                    .contains(features)
        })
        .map(|modifier| modifier.drm_format_modifier)
        .collect()
}
