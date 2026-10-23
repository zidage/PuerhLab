//! The image entry of the segmentation engine (SM3.C plan, decision C7).
//!
//! An image entry holds what the engine knows about one image: the image size, the image embedding
//! (SM3.B), the S3OD input image (C8), and the task results. The sessions and the cache entry of one image
//! share one entry through `Arc`. Each task result is computed once, at the first request that
//! needs it, and stays until the entry is dropped; the entry lifetime ends at the latest with an
//! unload or a model change, so a result never outlives the image or the model set that made it.
//!
//! Each result has its own mutex. The engine holds that mutex while it computes the result, so two
//! requests on one image compute it once. A failed computation stores nothing. The engine locks
//! the Sky and People results before the head result, never the reverse.

use std::sync::{Arc, Mutex};

use super::backend::{HeadOutput, ImageFeatures};
use super::task_masks::{PersonBox, SubjectAlpha, SubjectInputImage};

/// Sky result of one image (SM3.C plan, section 2.3).
#[derive(Debug, Clone, PartialEq)]
pub struct SkyResult {
    pub presence: f32,
    pub presence_threshold: f32,
    /// The selected decoder mask of each region, clipped logits; empty when no sky is found.
    pub region_logits: Vec<Vec<f32>>,
    /// The "not found" text; empty when the union exists.
    pub message: String,
}

/// One entry of the People list and the mask of that person.
#[derive(Debug, Clone, PartialEq)]
pub struct PersonMask {
    pub person: PersonBox,
    /// The selected decoder mask of the box prompt, clipped logits.
    pub logits: Vec<f32>,
}

/// People result of one image (SM3.C plan, section 2.4).
#[derive(Debug, Clone, PartialEq)]
pub struct PeopleResult {
    pub presence: f32,
    pub presence_threshold: f32,
    /// The People list in its list order (C3); empty when no people are found.
    pub people: Vec<PersonMask>,
    /// The "not found" text; empty when the list has people.
    pub message: String,
}

/// A task result slot of an image entry: empty until the first successful computation.
pub type ResultSlot<T> = Mutex<Option<Arc<T>>>;

/// See the module documentation.
pub struct ImageEntry {
    /// Size of the session image; the size of the Sky and People union masks.
    pub width: usize,
    pub height: usize,
    pub features: Arc<dyn ImageFeatures>,
    pub subject_input: SubjectInputImage,
    pub head: ResultSlot<HeadOutput>,
    pub sky: ResultSlot<SkyResult>,
    pub people: ResultSlot<PeopleResult>,
    pub subject: ResultSlot<SubjectAlpha>,
}

impl ImageEntry {
    pub fn new(
        width: usize,
        height: usize,
        features: Arc<dyn ImageFeatures>,
        subject_input: SubjectInputImage,
    ) -> Self {
        Self {
            width,
            height,
            features,
            subject_input,
            head: Mutex::new(None),
            sky: Mutex::new(None),
            people: Mutex::new(None),
            subject: Mutex::new(None),
        }
    }
}
