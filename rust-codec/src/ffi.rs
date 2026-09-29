//! C ABI for the Python `torch` backend switch (M4).
//!
//! ```c
//! void* cutetts_vae_load(const char* safetensors_path); // null on error
//! void  cutetts_vae_free(void* handle);
//! int   cutetts_vae_decode(void* h, const float* latent64xT, unsigned long frames, float* wav_out);
//! // wav_out must hold frames*1920 floats. Returns 0 ok, -1 error.
//! ```

use crate::decode::decode_nth;
use crate::weights::{load_decoder_weights, DecoderW};
use std::ffi::{c_char, c_int, c_ulong, CStr};

fn err() -> *mut DecoderW {
    std::ptr::null_mut()
}

/// Load + fuse weights. `CUTETTS_THREADS` controls worker count at decode time.
#[no_mangle]
pub extern "C" fn cutetts_vae_load(path: *const c_char) -> *mut DecoderW {
    if path.is_null() {
        return err();
    }
    let path = unsafe { CStr::from_ptr(path) }.to_string_lossy().into_owned();
    let result = std::panic::catch_unwind(|| load_decoder_weights(std::path::Path::new(&path)));
    match result {
        Ok(w) => Box::into_raw(Box::new(w)),
        Err(_) => err(),
    }
}

#[no_mangle]
pub extern "C" fn cutetts_vae_free(handle: *mut DecoderW) {
    if !handle.is_null() {
        unsafe {
            drop(Box::from_raw(handle));
        }
    }
}

#[no_mangle]
pub extern "C" fn cutetts_vae_decode(
    handle: *mut DecoderW,
    latent: *const f32,
    frames: c_ulong,
    wav_out: *mut f32,
) -> c_int {
    if handle.is_null() || latent.is_null() || wav_out.is_null() || frames == 0 {
        return -1;
    }
    let result = std::panic::catch_unwind(|| {
        let w = unsafe { &*handle };
        let frames = frames as usize;
        let latent = unsafe { std::slice::from_raw_parts(latent, 64 * frames) };
        let nth = crate::conv::default_threads();
        let wav = decode_nth(w, latent, frames, nth);
        assert_eq!(wav.len(), frames * crate::decoder::HOP_LENGTH);
        unsafe { std::ptr::copy_nonoverlapping(wav.as_ptr(), wav_out, wav.len()) };
    });
    if result.is_ok() {
        0
    } else {
        -1
    }
}
