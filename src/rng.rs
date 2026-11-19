//! An exact copy of `np.random.default_rng(seed)`: SeedSequence -> PCG64 and the distributions the
//! generator uses. This way a given seed produces the same layout as the numpy version did.

const INIT_A: u32 = 0x43b0_d7e5;
const MULT_A: u32 = 0x931e_8875;
const INIT_B: u32 = 0x8b51_f9dd;
const MULT_B: u32 = 0x58f3_8ded;
const MIX_MULT_L: u32 = 0xca01_f9dd;
const MIX_MULT_R: u32 = 0x4973_f715;
const XSHIFT: u32 = 16;
const POOL_SIZE: usize = 4;

const PCG_MULT: u128 = 0x2360_ED05_1FC6_5DA4_4385_DF64_9FCC_F645;

fn hashmix(value: u32, hash_const: &mut u32) -> u32 {
    let mut v = value ^ *hash_const;
    *hash_const = hash_const.wrapping_mul(MULT_A);
    v = v.wrapping_mul(*hash_const);
    v ^ (v >> XSHIFT)
}

fn mix(x: u32, y: u32) -> u32 {
    let r = MIX_MULT_L.wrapping_mul(x).wrapping_sub(MIX_MULT_R.wrapping_mul(y));
    r ^ (r >> XSHIFT)
}

/// `SeedSequence(seed).generate_state(4, np.uint64)`.
fn seed_state(seed: u64) -> [u64; 4] {
    // integer -> uint32 words, little-endian; 0 becomes [0]
    let mut entropy = vec![seed as u32];
    if seed >> 32 != 0 {
        entropy.push((seed >> 32) as u32);
    }
    let mut pool = [0u32; POOL_SIZE];
    let mut hc = INIT_A;
    for (i, slot) in pool.iter_mut().enumerate() {
        *slot = hashmix(entropy.get(i).copied().unwrap_or(0), &mut hc);
    }
    for src in 0..POOL_SIZE {
        for dst in 0..POOL_SIZE {
            if src != dst {
                pool[dst] = mix(pool[dst], hashmix(pool[src], &mut hc));
            }
        }
    }
    for &e in entropy.iter().skip(POOL_SIZE) {
        for slot in pool.iter_mut() {
            *slot = mix(*slot, hashmix(e, &mut hc));
        }
    }
    let mut words = [0u32; 8];
    let mut hc = INIT_B;
    for (i, w) in words.iter_mut().enumerate() {
        let mut v = pool[i % POOL_SIZE] ^ hc;
        hc = hc.wrapping_mul(MULT_B);
        v = v.wrapping_mul(hc);
        *w = v ^ (v >> XSHIFT);
    }
    std::array::from_fn(|k| words[2 * k] as u64 | (words[2 * k + 1] as u64) << 32)
}

pub struct Rng {
    state: u128,
    inc: u128,
    // PCG64.next_uint32 splits one 64-bit draw into two halves and keeps the second for later
    spare32: Option<u32>,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        let s = seed_state(seed);
        let init_state = (s[0] as u128) << 64 | s[1] as u128;
        let init_seq = (s[2] as u128) << 64 | s[3] as u128;
        let mut r = Rng { state: 0, inc: (init_seq << 1) | 1, spare32: None };
        r.step();
        r.state = r.state.wrapping_add(init_state);
        r.step();
        r
    }

    fn step(&mut self) {
        self.state = self.state.wrapping_mul(PCG_MULT).wrapping_add(self.inc);
    }

    pub fn next_u64(&mut self) -> u64 {
        self.step();
        let hi = (self.state >> 64) as u64;
        let lo = self.state as u64;
        (hi ^ lo).rotate_right((self.state >> 122) as u32)
    }

    fn next_u32(&mut self) -> u32 {
        if let Some(v) = self.spare32.take() {
            return v;
        }
        let n = self.next_u64();
        self.spare32 = Some((n >> 32) as u32);
        n as u32
    }

    /// `rng.random()`: 0..1
    pub fn random(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }

    pub fn random_vec(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.random()).collect()
    }

    /// `rng.uniform(low, high)`
    pub fn uniform(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.random()
    }

    /// `rng.integers(low, high, n)` for ranges that fit in 32 bits: Lemire's method on `next_uint32`,
    /// which splits 64-bit draws into halves (the buffer is shared across all calls).
    pub fn integers_vec(&mut self, low: i64, high: i64, n: usize) -> Vec<i64> {
        let rng = (high - low - 1) as u64;
        assert!(rng < u32::MAX as u64, "integers range exceeds 32 bits");
        if rng == 0 {
            return vec![low; n];
        }
        let excl = rng + 1;
        (0..n)
            .map(|_| {
                let mut m = self.next_u32() as u64 * excl;
                let mut left = m as u32 as u64;
                if left < excl {
                    let threshold = (u32::MAX as u64 - rng) % excl;
                    while left < threshold {
                        m = self.next_u32() as u64 * excl;
                        left = m as u32 as u64;
                    }
                }
                low + (m >> 32) as i64
            })
            .collect()
    }

    pub fn integers(&mut self, low: i64, high: i64) -> i64 {
        self.integers_vec(low, high, 1)[0]
    }

    /// `rng.permutation(arr)` for a 1-D array (Fisher-Yates with `random_interval`).
    pub fn permutation<T: Copy>(&mut self, arr: &[T]) -> Vec<T> {
        let mut out = arr.to_vec();
        for i in (1..out.len()).rev() {
            let j = self.interval(i as u64) as usize;
            out.swap(i, j);
        }
        out
    }

    fn interval(&mut self, max: u64) -> u64 {
        if max == 0 {
            return 0;
        }
        let mut mask = max;
        for s in [1, 2, 4, 8, 16, 32] {
            mask |= mask >> s;
        }
        if max <= u32::MAX as u64 {
            loop {
                let v = (self.next_u32() as u64) & mask;
                if v <= max {
                    return v;
                }
            }
        }
        loop {
            let v = self.next_u64() & mask;
            if v <= max {
                return v;
            }
        }
    }

    /// Normal distribution N(0, 1). numpy uses a ziggurat; this is Box-Muller: same distribution,
    /// different samples. Only used for film grain, so the difference is invisible.
    pub fn normal_vec(&mut self, n: usize) -> Vec<f64> {
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            let u1 = 1.0 - self.random();
            let u2 = self.random();
            let r = (-2.0 * u1.ln()).sqrt();
            let a = std::f64::consts::TAU * u2;
            out.push(r * a.cos());
            if out.len() < n {
                out.push(r * a.sin());
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_sequence() {
        assert_eq!(seed_state(1), [0x672d8ee56d6791ff, 0x8ae19ca14eb1072c, 0x4915796d1322fc4a, 0xd0cc2bdcaba049bd]);
    }

    #[test]
    fn raw() {
        for (seed, a, b) in [
            (1, 0x8306bdf37922e4ffu64, 0xf35196bbc152a866u64),
            (7, 0xa00641a9f1e54a8b, 0xe5afcdbcaf266a95),
            (123456, 0xa2f290ab1932e2c7, 0x6283046bd0d47d21),
        ] {
            let mut r = Rng::new(seed);
            assert_eq!((r.next_u64(), r.next_u64()), (a, b), "seed {seed}");
        }
    }

    #[test]
    fn distributions() {
        let mut r = Rng::new(1);
        assert_eq!(r.random_vec(3), vec![0.5118216247002567, 0.9504636963259353, 0.14415961271963373]);
        let mut r = Rng::new(1);
        assert_eq!(r.integers(0, 10_000), 4731);
        assert_eq!(r.integers_vec(0, 1920, 5), vec![982, 1449, 1824, 66, 276]);
        assert_eq!((r.uniform(0.0, 6.0), r.uniform(0.0, 6.0)), (5.691896682823463, 1.8709887120629127));
        let mut r = Rng::new(1);
        assert_eq!(r.permutation(&[0, 1, 2, 3, 4, 0]), vec![4, 0, 2, 1, 0, 3]);
        assert_eq!(r.uniform(0.0, 6.0), 1.8709887120629127);
        let mut r = Rng::new(1);
        r.permutation(&[0, 1, 2, 3, 4, 5]);
        assert_eq!(r.integers_vec(0, 100, 3), vec![94, 24, 31]);
        assert_eq!(r.permutation(&[0, 1, 2, 3]), vec![2, 3, 1, 0]);
    }
}
