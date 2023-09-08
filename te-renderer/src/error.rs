use std::fmt::Display;

#[derive(Debug)]
pub enum TError {
    UninitializedModel,
    GLBModelLoadingFail,
    SpriteLoadingFail,
    UninitializedSprite,
    EmptySpriteArray,
    NamelessGLB,
    InvalidGLB,
    SizeRequired,
}

impl Display for TError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TError::UninitializedModel => write!(f, "UninitializedModel"),
            TError::GLBModelLoadingFail => write!(f, "GLBModelLoadingFail"),
            TError::SpriteLoadingFail => write!(f, "SpriteLoadingFail"),
            TError::UninitializedSprite => write!(f, "UninitializedSprite"),
            TError::EmptySpriteArray => write!(f, "EmptySpriteArray"),
            TError::NamelessGLB => write!(f, "NamelessGLB"),
            TError::InvalidGLB => write!(f, "InvalidGLB"),
            TError::SizeRequired => write!(f, "SizeRequired"),
        }
    }
}

impl std::error::Error for TError {}
