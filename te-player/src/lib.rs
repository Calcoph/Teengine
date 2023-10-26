#[cfg(feature = "imgui")]
pub mod imgui;
mod raw;

use std::{error::Error, fmt::Display};

pub use crate::raw::*;

#[derive(Debug)]
enum InitError {
    Unkown,
}

impl Display for InitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let msg = match self {
            InitError::Unkown => "InitError: Unkown error",
        };

        write!(f, "{}", msg)
    }
}

impl Error for InitError {}
