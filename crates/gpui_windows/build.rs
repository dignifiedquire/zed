#![allow(clippy::disallowed_methods, reason = "build scripts are exempt")]

fn main() {
    #[cfg(not(all(target_os = "windows", not(debug_assertions))))]
    println!("cargo:rerun-if-changed=build.rs");
    #[cfg(target_os = "windows")]
    {
        // Compile HLSL shaders
        #[cfg(not(debug_assertions))]
        compile_shaders();
    }
    #[cfg(not(target_os = "windows"))]
    prebuilt_shaders::copy();
}

/// Returns the FNV-1a hash of `bytes`.
#[cfg(any(not(target_os = "windows"), not(debug_assertions)))]
fn source_hash(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// Returns the first line of a compiled shader file, which names the hash of
/// the `shaders.hlsl` it was compiled from.
#[cfg(any(not(target_os = "windows"), not(debug_assertions)))]
fn source_line(source: &[u8]) -> String {
    format!("// shaders.hlsl FNV-1a {:016x}\n", source_hash(source))
}

/// Cross-compiled Windows release builds take their shaders from a native
/// build, as fxc.exe runs only on Windows.
#[cfg(not(target_os = "windows"))]
mod prebuilt_shaders {
    use std::{env, fs, path::PathBuf, process};

    pub fn copy() {
        let is_windows_release = env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows")
            && env::var_os("CARGO_CFG_DEBUG_ASSERTIONS").is_none();
        if !is_windows_release {
            return;
        }
        println!("cargo:rerun-if-env-changed=GPUI_SHADERS_BYTES");
        let shader_path =
            PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("src/shaders.hlsl");
        println!("cargo:rerun-if-changed={}", shader_path.display());
        let Some(prebuilt) = env::var_os("GPUI_SHADERS_BYTES").map(PathBuf::from) else {
            println!(
                "cargo::error=A Windows release build cross-compiled from another OS needs \
                 GPUI_SHADERS_BYTES: the shaders_bytes.rs a native Windows release build of the \
                 same shaders.hlsl wrote to its OUT_DIR."
            );
            process::exit(1);
        };
        println!("cargo:rerun-if-changed={}", prebuilt.display());
        let source = fs::read(&shader_path).expect("Failed to read shaders.hlsl");
        let compiled = fs::read_to_string(&prebuilt).expect("Failed to read GPUI_SHADERS_BYTES");
        if !compiled.starts_with(&super::source_line(&source)) {
            println!(
                "cargo::error={} was compiled from another shaders.hlsl. Build gpui_windows \
                 natively on Windows in release and take its OUT_DIR/shaders_bytes.rs.",
                prebuilt.display()
            );
            process::exit(1);
        }
        let out = PathBuf::from(env::var("OUT_DIR").unwrap()).join("shaders_bytes.rs");
        fs::write(out, compiled).expect("Failed to write shaders_bytes.rs");
    }
}

#[cfg(all(target_os = "windows", not(debug_assertions)))]
mod shader_compilation {
    use std::{
        fs,
        io::Write,
        path::{Path, PathBuf},
        process::{self, Command},
    };

    pub fn compile_shaders() {
        let shader_path =
            PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("src/shaders.hlsl");
        let out_dir = std::env::var("OUT_DIR").unwrap();

        println!("cargo:rerun-if-changed={}", shader_path.display());

        // Check if fxc.exe is available
        let fxc_path = find_fxc_compiler();

        // Define all modules
        let modules = [
            "quad",
            "opaque_quad",
            "shadow",
            "path_tile",
            "underline",
            "monochrome_sprite",
            "subpixel_sprite",
            "polychrome_sprite",
        ];

        let rust_binding_path = format!("{}/shaders_bytes.rs", out_dir);
        // A cross-compiled build checks this line against its own shaders.hlsl.
        let source = fs::read(&shader_path).expect("Failed to read shaders.hlsl");
        fs::write(&rust_binding_path, super::source_line(&source))
            .expect("Failed to start the Rust binding file");
        for module in modules {
            compile_shader_for_module(
                module,
                &out_dir,
                &fxc_path,
                shader_path.to_str().unwrap(),
                &rust_binding_path,
            );
        }
    }

    /// Locate `binary` in the newest installed Windows SDK.
    pub fn find_latest_windows_sdk_binary(
        binary: &str,
    ) -> Result<Option<PathBuf>, Box<dyn std::error::Error>> {
        let key = windows_registry::LOCAL_MACHINE
            .open("SOFTWARE\\WOW6432Node\\Microsoft\\Microsoft SDKs\\Windows\\v10.0")?;

        let install_folder: String = key.get_string("InstallationFolder")?; // "C:\Program Files (x86)\Windows Kits\10\"
        let install_folder_bin = Path::new(&install_folder).join("bin");

        let mut versions: Vec<_> = std::fs::read_dir(&install_folder_bin)?
            .flatten()
            .filter(|entry| entry.path().is_dir())
            .filter_map(|entry| entry.file_name().into_string().ok())
            .collect();

        versions.sort_by_key(|s| {
            s.split('.')
                .filter_map(|p| p.parse().ok())
                .collect::<Vec<u32>>()
        });

        let arch = match std::env::consts::ARCH {
            "x86_64" => "x64",
            "aarch64" => "arm64",
            _ => Err(format!(
                "Unsupported architecture: {}",
                std::env::consts::ARCH
            ))?,
        };

        if let Some(highest_version) = versions.last() {
            return Ok(Some(
                install_folder_bin
                    .join(highest_version)
                    .join(arch)
                    .join(binary),
            ));
        }

        Ok(None)
    }

    /// You can set the `GPUI_FXC_PATH` environment variable to specify the path to the fxc.exe compiler.
    fn find_fxc_compiler() -> String {
        // Check environment variable
        if let Ok(path) = std::env::var("GPUI_FXC_PATH")
            && Path::new(&path).exists()
        {
            return path;
        }

        // Try to find in PATH
        // NOTE: This has to be `where.exe` on Windows, not `where`, it must be ended with `.exe`
        if let Ok(output) = std::process::Command::new("where.exe")
            .arg("fxc.exe")
            .output()
            && output.status.success()
        {
            let path = String::from_utf8_lossy(&output.stdout);
            return path.trim().to_string();
        }

        if let Ok(Some(path)) = find_latest_windows_sdk_binary("fxc.exe") {
            return path.to_string_lossy().into_owned();
        }

        panic!("Failed to find fxc.exe");
    }

    fn compile_shader_for_module(
        module: &str,
        out_dir: &str,
        fxc_path: &str,
        shader_path: &str,
        rust_binding_path: &str,
    ) {
        // Compile vertex shader
        let output_file = format!("{}/{}_vs.h", out_dir, module);
        let const_name = format!("{}_VERTEX_BYTES", module.to_uppercase());
        compile_shader_impl(
            fxc_path,
            &format!("{module}_vertex"),
            &output_file,
            &const_name,
            shader_path,
            "vs_4_1",
        );
        generate_rust_binding(&const_name, &output_file, rust_binding_path);

        // Compile fragment shader
        let output_file = format!("{}/{}_ps.h", out_dir, module);
        let const_name = format!("{}_FRAGMENT_BYTES", module.to_uppercase());
        compile_shader_impl(
            fxc_path,
            &format!("{module}_fragment"),
            &output_file,
            &const_name,
            shader_path,
            "ps_4_1",
        );
        generate_rust_binding(&const_name, &output_file, rust_binding_path);
    }

    fn compile_shader_impl(
        fxc_path: &str,
        entry_point: &str,
        output_path: &str,
        var_name: &str,
        shader_path: &str,
        target: &str,
    ) {
        let output = Command::new(fxc_path)
            .args([
                "/T",
                target,
                "/E",
                entry_point,
                "/Fh",
                output_path,
                "/Vn",
                var_name,
                "/O3",
                shader_path,
            ])
            .output();

        match output {
            Ok(result) => {
                if result.status.success() {
                    return;
                }
                println!(
                    "cargo::error=Shader compilation failed for {}:\n{}",
                    entry_point,
                    String::from_utf8_lossy(&result.stderr)
                );
                process::exit(1);
            }
            Err(e) => {
                println!("cargo::error=Failed to run fxc for {}: {}", entry_point, e);
                process::exit(1);
            }
        }
    }

    fn generate_rust_binding(const_name: &str, head_file: &str, output_path: &str) {
        let header_content = fs::read_to_string(head_file).expect("Failed to read header file");
        let const_definition = {
            let global_var_start = header_content.find("const BYTE").unwrap();
            let global_var = &header_content[global_var_start..];
            let equal = global_var.find('=').unwrap();
            global_var[equal + 1..].trim()
        };
        let rust_binding = format!(
            "const {}: &[u8] = &{}\n",
            const_name,
            const_definition.replace('{', "[").replace('}', "]")
        );
        let mut options = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(output_path)
            .expect("Failed to open Rust binding file");
        options
            .write_all(rust_binding.as_bytes())
            .expect("Failed to write Rust binding file");
    }
}

#[cfg(all(target_os = "windows", not(debug_assertions)))]
use shader_compilation::compile_shaders;
