//! Compiled core of nestmodel: color refinement (`wl`) and edge rewiring (`rewire`).
use pyo3::prelude::*;

mod extras;
mod rewire;
mod sorting;
mod wl;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    wl::register(m)?;
    rewire::register(m)?;
    sorting::register(m)?;
    extras::register(m)?;
    Ok(())
}
