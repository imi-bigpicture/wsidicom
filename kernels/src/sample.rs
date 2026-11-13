//! The sample types the kernels take (uint8, uint16, float32), the RGB
//! pixel size, and the Q15 fixed-point weights of the uint8 kernels.

/// Samples per pixel of an RGB image.
pub(crate) const RGB: usize = 3;

/// A resample weight in Q15, the fixed-point format `mulhrs` multiplies by.
/// Clamped: a weight of exactly 1.0 would round to one past `i16::MAX`.
/// Always inlined: inside the SIMD kernels `round()` then compiles to
/// SSE4.1 instructions instead of a libm call.
#[inline(always)]
pub(crate) fn q15(weight: f32) -> i16 {
    ((weight * 32768.0).round() as i32).clamp(-32768, 32767) as i16
}

/// Scalar `mulhrs`: `(value * weight) >> 15` rounded to nearest — the multiply the
/// fixed-point u8 resample kernels are built on.
#[inline]
pub(crate) fn mulhrs(value: i32, weight: i16) -> i32 {
    (value * i32::from(weight) + (1 << 14)) >> 15
}

/// Fraction bits u8 pixels carry through the convolution's i16 rows:
/// Q6 leaves headroom for Catmull-Rom's ~1.14x overshoot, which would push
/// Q7 (255 << 7 = 32640) past i16.
pub(crate) const CONVOLVE_U8_FRAC_BITS: u32 = 6;

/// Panics unless `src` holds a `width` x `height` image of `channels` (1 or
/// 3) samples per pixel: the layout the kernels' unchecked loads rely on.
pub(crate) fn assert_image<T>(src: &[T], width: usize, height: usize, channels: usize) {
    assert!(
        matches!(channels, 1 | RGB),
        "images have 1 or 3 channels, not {channels}"
    );
    let samples = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(channels));
    assert!(
        samples.is_some_and(|samples| src.len() >= samples),
        "the image holds fewer than width * height * channels samples"
    );
}

/// Panics unless `channels` is the interleaved layout, RGB.
pub(crate) fn assert_interleaved(channels: usize) {
    assert!(
        channels == RGB,
        "interleaved images have 3 channels, not {channels}"
    );
}

/// A sample type: its value as f32 and back, rounded and saturated.
pub(crate) trait Sample: Copy + numpy::Element {
    fn to_f32(self) -> f32;

    fn from_f32(value: f32) -> Self;
}

impl Sample for u8 {
    fn to_f32(self) -> f32 {
        f32::from(self)
    }
    fn from_f32(value: f32) -> Self {
        // Branchless round+clamp: Rust's float->int cast saturates to [0,255],
        // so (value + 0.5) as u8 vectorizes (no round()/clamp() calls to block SIMD).
        (value + 0.5) as u8
    }
}

impl Sample for u16 {
    fn to_f32(self) -> f32 {
        f32::from(self)
    }
    fn from_f32(value: f32) -> Self {
        (value + 0.5) as u16
    }
}

impl Sample for f32 {
    fn to_f32(self) -> f32 {
        self
    }
    fn from_f32(value: f32) -> Self {
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q15_rounds_to_nearest_and_clamps_to_i16() {
        assert_eq!(q15(0.0), 0);
        assert_eq!(q15(0.5), 16384);
        assert_eq!(q15(-0.5), -16384);
        assert_eq!(q15(-1.0), -32768);
        // 1.0 would be 32768, one past i16::MAX.
        assert_eq!(q15(1.0), i16::MAX);
        assert_eq!(q15(2.5), i16::MAX);
        assert_eq!(q15(-2.5), i16::MIN);
        // Halfway between steps rounds away from zero.
        assert_eq!(q15(0.5 / 32768.0), 1);
        assert_eq!(q15(-0.5 / 32768.0), -1);
        assert_eq!(q15(1.5 / 32768.0), 2);
        assert_eq!(q15(0.49 / 32768.0), 0);
    }

    #[test]
    fn mulhrs_rounds_the_q15_product_to_nearest() {
        assert_eq!(mulhrs(0, i16::MAX), 0);
        assert_eq!(mulhrs(255 << 7, q15(0.5)), 255 << 6);
        assert_eq!(mulhrs(1, 16384), 1);
        assert_eq!(mulhrs(-1, 16384), 0);
        assert_eq!(mulhrs(32767, 32767), 32766);
        assert_eq!(mulhrs(-32768, 32767), -32767);
        // Unlike pmulhrsw, the i32 result does not wrap.
        assert_eq!(mulhrs(-32768, -32768), 32768);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn mulhrs_equals_pmulhrsw_on_i16_operands() {
        use crate::testing::Rng;
        use std::arch::x86_64::*;
        if !std::is_x86_feature_detected!("ssse3") {
            return;
        }
        let edges: [i16; 12] = [
            i16::MIN,
            i16::MIN + 1,
            -16385,
            -16384,
            -2,
            -1,
            0,
            1,
            2,
            16384,
            i16::MAX - 1,
            i16::MAX,
        ];
        let mut rng = Rng::new(80);
        let random = (0..4096).map(|_| {
            let [value_low, value_high, weight_low, weight_high, ..] = rng.next_u64().to_le_bytes();
            (
                i16::from_le_bytes([value_low, value_high]),
                i16::from_le_bytes([weight_low, weight_high]),
            )
        });
        let pairs = edges
            .iter()
            .flat_map(|&value| edges.iter().map(move |&weight| (value, weight)))
            .chain(random)
            // The one product pmulhrsw wraps (to i16::MIN).
            .filter(|&pair| pair != (i16::MIN, i16::MIN));
        for (value, weight) in pairs {
            // SAFETY: SSSE3 is available.
            let simd = unsafe {
                _mm_extract_epi16::<0>(_mm_mulhrs_epi16(
                    _mm_set1_epi16(value),
                    _mm_set1_epi16(weight),
                ))
            };
            let scalar = mulhrs(i32::from(value), weight);
            assert_eq!(i32::from(simd as i16), scalar, "{value} * {weight}");
        }
    }

    #[test]
    fn integer_samples_round_half_up_and_saturate() {
        assert_eq!(u8::from_f32(0.49), 0);
        assert_eq!(u8::from_f32(0.5), 1);
        assert_eq!(u8::from_f32(254.5), 255);
        assert_eq!(u8::from_f32(300.0), 255);
        assert_eq!(u8::from_f32(-3.0), 0);
        assert_eq!(u8::from_f32(f32::NAN), 0);
        assert_eq!(u16::from_f32(65534.5), u16::MAX);
        assert_eq!(u16::from_f32(1e9), u16::MAX);
        assert_eq!(u16::from_f32(-0.6), 0);
        assert_eq!(f32::from_f32(-0.25), -0.25);
        assert_eq!(u16::MAX.to_f32(), 65535.0);
    }

    #[test]
    fn assert_image_accepts_slices_holding_the_image() {
        assert_image::<u8>(&[], 0, 5, 1);
        assert_image(&[0u8; 12], 2, 2, RGB);
        assert_image(&[0u16; 13], 2, 2, RGB);
        assert_image(&[0f32; 6], 3, 2, 1);
    }

    #[test]
    #[should_panic(expected = "images have 1 or 3 channels, not 2")]
    fn assert_image_rejects_two_channels() {
        assert_image(&[0u8; 64], 2, 2, 2);
    }

    #[test]
    #[should_panic(expected = "images have 1 or 3 channels, not 4")]
    fn assert_image_rejects_four_channels() {
        assert_image(&[0u8; 64], 2, 2, 4);
    }

    #[test]
    #[should_panic(expected = "images have 1 or 3 channels, not 0")]
    fn assert_image_rejects_no_channels() {
        assert_image::<u8>(&[], 0, 0, 0);
    }

    #[test]
    #[should_panic(expected = "fewer than width * height * channels samples")]
    fn assert_image_rejects_a_short_slice() {
        assert_image(&[0u8; 11], 2, 2, RGB);
    }

    #[test]
    #[should_panic(expected = "fewer than width * height * channels samples")]
    fn assert_image_rejects_pixels_overflowing_usize() {
        assert_image(&[0u8; 16], usize::MAX, 2, 1);
    }

    #[test]
    #[should_panic(expected = "fewer than width * height * channels samples")]
    fn assert_image_rejects_samples_overflowing_usize() {
        assert_image(&[0u8; 16], usize::MAX / 2, 2, RGB);
    }

    #[test]
    fn assert_interleaved_accepts_rgb() {
        assert_interleaved(RGB);
    }

    #[test]
    #[should_panic(expected = "interleaved images have 3 channels, not 1")]
    fn assert_interleaved_rejects_planar() {
        assert_interleaved(1);
    }
}
