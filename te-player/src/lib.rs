mod raw;
#[cfg(feature = "imgui-0-8")]
pub mod imgui;

use std::{fmt::Display, error::Error};

pub use crate::raw::*;

#[derive(Debug)]
enum InitError {
    Unkown,
    Opaque
}

impl Display for InitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let msg = match self {
            InitError::Opaque => "InitError: icon.png must be a transparent png",
            InitError::Unkown => "InitError: Unkown error",
        };

        write!(f, "{}", msg)
    }
}

impl Error for InitError {}
