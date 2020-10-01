use core::time::Duration;
use pbr::ProgressBar;
use rand::Rng;
use rayon::prelude::*;
use std::cell::UnsafeCell;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::Mutex;
use std::time::Instant;

const FILE_PATH: &str = "prova.csv";

const L: usize = 4;
const D: usize = 3;

#[derive(Debug)]
struct MetroParams {
    delta: f64,
    ntherm: i32,
    nsweep: i32,
    naccu: i32,
}

#[derive(Debug)]
struct ActionParams {
    kappa: f64,
    lambda: f64,
}

#[derive(Debug)]
struct Dimensions {
    l: usize,
    d: usize,
    v: usize,
}

impl Dimensions {
    fn new(d: usize, l: usize) -> Self {
        Dimensions {
            l: l,
            d: d,
            v: l.pow(d as u32),
        }
    }
}

#[derive(Debug)]
struct Fields<'a> {
    dim: &'a Dimensions,
    act_params: &'a ActionParams,
    phi: UnsafeCell<[f64; 100]>,
    hop: Vec<Vec<usize>>,
}

unsafe impl Sync for Fields<'_> {}

macro_rules! my_phi {
    ($phi:expr, $index:expr) => {
        unsafe { (*$phi.get())[$index] }
    };
}

impl<'a> Fields<'a> {
    fn new(dimensions: &'a Dimensions, action_params: &'a ActionParams) -> Self {
        let mut rng = rand::thread_rng();
        let mut phi = [0.0; 100];

        for i in 0..dimensions.v {
            phi[i] = rng.gen();
        }

        Fields {
            dim: dimensions,
            act_params: action_params,
            phi: UnsafeCell::new(phi),
            hop: vec![vec![0; 2 * dimensions.d]; dimensions.v],
        }
    }

    fn hopping(mut self) -> Self {
        let l = self.dim.l as i32;
        let v = self.dim.v as i32;
        let d = self.dim.d as i32;

        for x in 0..v as usize {
            let mut lk = v;
            let mut y = x as i32;
            let mut xk: i32;
            let mut dxk: i32;

            for k in (0..d as usize).rev() {
                lk /= l;
                xk = y / lk;
                y = y - xk * lk;

                if xk < l - 1 {
                    dxk = lk;
                } else {
                    dxk = lk * (1 - l)
                }
                self.hop[x][k] = (x as i32 + dxk) as usize;

                if xk > 0 {
                    dxk = -lk;
                } else {
                    dxk = lk * (l - 1);
                }
                self.hop[x][k + d as usize] = (x as i32 + dxk) as usize;
            }
        }
        self
    }

    #[allow(dead_code)]
    fn action(&self) -> f64 {
        let d = self.dim.d;
        let v = self.dim.v;

        let mut action_result = 0.0;

        for i in 0..v {
            let mut phin = 0.0;
            for mu in 0..d {
                phin += my_phi!(self.phi, self.hop[i][mu]);
            }

            let phi2 = my_phi!(self.phi, i).powf(2.0);
            action_result += -2.0 * self.act_params.kappa * phin * my_phi!(self.phi, i)
                + phi2
                + self.act_params.lambda * (phi2 - 1.0).powf(2.0);
        }

        action_result
    }

    fn delta_s(&self, metro_params: &MetroParams, index: usize, r: f64) -> f64 {
        let d = self.dim.d;
        let kappa = self.act_params.kappa;
        let lambda = self.act_params.lambda;
        let delta = metro_params.delta;

        let p = my_phi!(self.phi, index);
        let p2 = p.powi(2);
        let p3 = p.powi(3);

        let f = delta * (r - 0.5);
        let f2 = f.powi(2);
        let f3 = f.powi(3);
        let f4 = f.powi(4);

        let mut mu_sum = 0.0;

        for mu in 0..d {
            mu_sum += my_phi!(self.phi, self.hop[index][mu]);
            mu_sum += my_phi!(self.phi, self.hop[index][mu + d]);
        }
        mu_sum *= -2.0 * f * kappa;

        mu_sum
            + 2.0 * p * f
            + f2
            + lambda * (4.0 * p3 * f + 6.0 * p2 * f2 + 4.0 * p * f3 - 4.0 * p * f + f4 - 2.0 * f2)
    }

    fn m(&self) -> f64 {
        unsafe {
            let phi = self.phi.get();
            (*phi).iter().take(self.dim.d).sum()
        }
    }
}

fn print_params(action_params: &ActionParams, metro_params: &MetroParams) {
    println!("PARAMETERS:");
    println!("{:<10}{}", "kappa", action_params.kappa);
    println!("{:<10}{}", "lambda", action_params.lambda);
    println!("{:<10}{}", "delta", metro_params.delta);
    println!("{:<10}{:e}", "nsweep", metro_params.nsweep);
    println!("{:<10}{:e}", "ntherm", metro_params.ntherm);
    println!("{:<10}{:e}", "naccu", metro_params.naccu);
}

fn metropolis(
    dim: &Dimensions,
    metro_params: &MetroParams,
    action_params: &ActionParams,
    file_path: &str,
) -> f64 {
    let fields = Fields::new(&dim, &action_params).hopping();

    let mut rng = rand::thread_rng();

    let ntherm = metro_params.ntherm;
    let nsweep = metro_params.nsweep;
    let naccu = metro_params.naccu;
    let delta = metro_params.delta;
    let kappa = action_params.kappa;
    let lambda = action_params.lambda;

    let v = fields.dim.v;
    let mut n_prop = 0;
    let mut n_acc = 0;

    let n_m = 0;

    let mut progress_bar = ProgressBar::new(ntherm as u64);
    progress_bar.set_width(Some(100));
    progress_bar.show_counter = false;
    progress_bar.show_tick = false;
    progress_bar.show_speed = false;
    progress_bar.message("Therm: ");
    progress_bar.show_message = true;
    let refresh_rate = Duration::from_millis(125);
    progress_bar.set_max_refresh_rate(Some(refresh_rate));

    // THERMALIZATION
    for _ in 0..ntherm {
        for i in 0..v {
            n_prop += 1;
            let r = rng.gen::<f64>();
            let exp_ds = (-fields.delta_s(&metro_params, i, r)).exp();

            if exp_ds >= 1.0 || rng.gen::<f64>() < exp_ds {
                unsafe {
                    // Data races will occurr, but should improve the overaa randomness :-)
                    (*fields.phi.get())[i] += delta * (r - 0.5);
                }
                n_acc += 1;
            }
        }
        progress_bar.inc();
    }

    println!();

    // SWEEP
    assert_eq!(nsweep % naccu, 0);
    let chunks = nsweep / naccu;
    progress_bar = ProgressBar::new(chunks as u64);
    progress_bar.set_width(Some(103));
    progress_bar.show_counter = false;
    progress_bar.show_tick = false;
    progress_bar.show_speed = false;
    progress_bar.message("Sweep: ");
    progress_bar.show_message = true;

    // open file
    let f = match File::create(file_path) {
        Err(why) => panic!("Could't open or create file '{}': {}", file_path, why),
        Ok(file) => file,
    };
    let mut f = BufWriter::new(f);

    //write file header
    f.write_all(
        format!(
            "{},{},{},{},{},{},{}\n",
            v, naccu, ntherm, nsweep, delta, kappa, lambda
        )
        .as_bytes(),
    )
    .expect("error writing header to file");
    f.write_all(b"n,M,absM,M2,M4\n")
        .expect("error writing header to file");

    let mbar = Mutex::new(progress_bar);
    n_prop += (nsweep as usize) * v;
    let output: Vec<(String, usize)> = (0..chunks)
        .into_par_iter()
        .map(move |_i| {
            let mut avg_m = 0.0;
            let mut avg_abs = 0.0;
            let mut avg_m2 = 0.0;
            let mut avg_m4 = 0.0;
            let mut rng = rand::thread_rng();
            let mut my_n_acc = 0;
            for _j in 0..naccu {
                for i in 0..v {
                    let r = rng.gen::<f64>();
                    let exp_ds = (-fields.delta_s(&metro_params, i, r)).exp();

                    if exp_ds >= 1.0 || rng.gen::<f64>() < exp_ds {
                        unsafe {
                            // Data races will occurr, but should improve the overaa randomness :-)
                            (*fields.phi.get())[i] += delta * (r - 0.5);
                        }
                        my_n_acc += 1;
                    }
                }
                let temp_m = fields.m();
                avg_m += temp_m;
                avg_abs += temp_m.abs();
                avg_m2 += temp_m.powi(2);
                avg_m4 += temp_m.powi(4);
            }
            mbar.lock().unwrap().inc();
            (
                format!("{},{},{},{},{}\n", n_m, avg_m, avg_abs, avg_m2, avg_m4),
                my_n_acc,
            )
        })
        .collect();
    for (s, mn) in output {
        f.write_all(s.as_bytes()).expect("error writing file");
        n_acc += mn;
    }

    println!();
    n_acc as f64 / n_prop as f64
}

fn main() {
    let metro_params = MetroParams {
        delta: 0.25,
        ntherm: 100_000,
        nsweep: 100_000_000,
        naccu: 10_000,
    };
    let action_params = ActionParams {
        kappa: 0.15,
        lambda: 1.145,
    };
    print_params(&action_params, &metro_params);
    println!();

    let dim = Dimensions::new(D, L);

    let start = Instant::now();
    let acc = metropolis(&dim, &metro_params, &action_params, FILE_PATH);
    let duration = start.elapsed();

    let seconds = duration.as_secs() % 60;
    let minutes = (duration.as_secs() / 60) % 60;
    let hours = (duration.as_secs() / 60) / 60;
    let time_str = format!("{}:{}:{}", hours, minutes, seconds);
    println!("Metropolis completed in {}!", time_str);
    println!("Accuracy = {:.2}%", acc * 100.0);
}
