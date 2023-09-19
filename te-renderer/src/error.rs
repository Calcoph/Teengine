use std::fmt::Display;

#[derive(Debug)]
pub enum GLBErr {
    // tex coords should be F32
    InvalidTexCoordDataType,
    // u8 and U16 is in the gltf spec but not supported in teengine
    UnsupportedTexCoordDataType,
    // tex coords should be Vec2
    InvalidTexCoordAccessorDimension,
    TODO
}

#[derive(Debug)]
pub enum TError {
    UninitializedModel,
    GLBModelLoadingFail,
    SpriteLoadingFail,
    UninitializedSprite,
    EmptySpriteArray,
    NamelessGLB,
    InvalidGLB(GLBErr),
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
            TError::InvalidGLB(_) => write!(f, "InvalidGLB"),
            TError::SizeRequired => write!(f, "SizeRequired"),
        }
    }
}

impl std::error::Error for TError {}
