//! CuteTTS AudioVAE decoder topology (M0 manifest).
//!
//! Config: vae_dim=64, decoder_dim=1536, rates=[16,8,5,3], depthwise=true,
//! product 16*8*5*3=1920 = 24000/12.5 hop.
//! Weights: model/CuteTTS/weights/audio_vae/model.safetensors (fp32, 243 tensors,
//! total 127.9M, decoder.* 122 tensors / 45.8M, encoder 82.1M).
//! Biggest: decoder.model.2.block.1.weight_v [1536,768,32] 37.75M
//! (ConvTranspose k=32 stride=16).

pub const VAE_DIM: usize = 64;
pub const DECODER_DIM: usize = 1536;
pub const DECODER_RATES: [usize; 4] = [16, 8, 5, 3];
pub const HOP_LENGTH: usize = 1920;

/// Decoder stage channels: 1536 -> 768 -> 384 -> 192 -> 96 -> 1
pub const STAGE_CHANNELS: [usize; 6] = [1536, 768, 384, 192, 96, 1];

/// Reference test vector: rust-codec/testdata/latent_1x64x8.npy
/// -> ref_wav_1x1x15360.npy (8 frames * 1920). M2 gate: max_abs_err < 1e-4.
pub const TEST_LATENT_FRAMES: usize = 8;
pub const TEST_WAV_SAMPLES: usize = 15360;
