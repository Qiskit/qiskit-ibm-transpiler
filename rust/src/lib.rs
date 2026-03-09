use pyo3::prelude::*;
pub mod routing;

use crate::routing::routing::CircuitRouting;

/// A Python module implemented in Rust.
#[pymodule]
fn qiskit_ibm_transpiler_rs(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CircuitRouting>()?;
    Ok(())
}
