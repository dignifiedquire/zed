//! Video frames in DMA-BUFs, imported into wgpu's Vulkan backend as the luma and chroma
//! textures of the surface shader.

use std::sync::Arc;

use anyhow::{Context as _, Result, anyhow, ensure};
use gpui::{DmaBufFrame, DmaBufPlane};

/// How many drawn frames an import outlives its last draw, so that a frame the GPU may
/// still sample stays alive and unchanged.
const KEPT_FRAMES: u64 = 3;

/// The imports of the frames that surfaces drew lately.
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
        frame: &Arc<DmaBufFrame>,
    ) -> Option<(wgpu::TextureView, wgpu::TextureView)> {
        match self.import(device, frame) {
            Ok(textures) => Some(textures),
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
        let luma = import_plane(
            device,
            &frame.luma,
            wgpu::TextureFormat::R8Unorm,
            frame.width,
            frame.height,
        )?;
        let chroma = import_plane(
            device,
            &frame.chroma,
            wgpu::TextureFormat::Rg8Unorm,
            frame.width.div_ceil(2),
            frame.height.div_ceil(2),
        )?;
        self.frames.push(ImportedFrame {
            frame: frame.clone(),
            luma: luma.clone(),
            chroma: chroma.clone(),
            last_drawn: self.drawn,
        });
        Ok((luma, chroma))
    }
}

/// Imports `plane` as a texture of `format`, `width` by `height` texels.
fn import_plane(
    device: &wgpu::Device,
    plane: &DmaBufPlane,
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
) -> Result<wgpu::TextureView> {
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
    // SAFETY: the texture comes from this device with `hal_descriptor`, which matches
    // `descriptor`, and the producer wrote its texels before handing the frame over.
    let texture = unsafe {
        device.create_texture_from_hal::<wgpu::hal::api::Vulkan>(
            texture,
            &descriptor,
            wgpu::TextureUses::RESOURCE,
            true,
        )
    };
    Ok(texture.create_view(&wgpu::TextureViewDescriptor::default()))
}
