pub mod event_loop;

use std::{io::Error, cell::RefCell, rc::Rc};
use te_renderer::{initial_config::InitialConfiguration, state::{TeState, GpuState}};
use te_winit::event_loop::EventLoopBuilder;
pub use winit as te_winit;
use winit::{window::{WindowBuilder, Icon, Window}, dpi, event_loop::EventLoop};
use image::io::Reader as ImageReader;

use te_gamepad::gamepad::{self, ControllerEvent};

/// Get all the structs needed to start the engine, skipping the boilerplate.
pub async fn prepare(config: InitialConfiguration, log: bool) -> Result<(EventLoop<ControllerEvent>, Rc<RefCell<GpuState>>, Rc<RefCell<Window>>, Rc<RefCell<TeState>>), Error> {
    if log {
        env_logger::init();
    }

    let img = match ImageReader::open(&config.icon_path)?.decode() {
        Ok(img) => img.to_rgba8(),
        Err(_) => panic!("Icon has wrong format"),
    };

    let event_loop = EventLoopBuilder::with_user_event().build();
    gamepad::listen(event_loop.create_proxy());

    let wb = WindowBuilder::new()
        .with_title(&config.window_name)
        .with_inner_size(dpi::LogicalSize::new(config.screen_width, config.screen_height))
        .with_window_icon(Some(match Icon::from_rgba(img.into_raw(), 64, 64) {
            Ok(icon) => icon,
            Err(_) => panic!("Couldn't get icon raw data")
        }));

    let window = wb.build(&event_loop).expect("Unable to create window");

    let gpu = GpuState::new(window.inner_size(), &window).await;
    let state = TeState::new(&window, &gpu, config).await;

    Ok((event_loop, Rc::new(RefCell::new(gpu)), Rc::new(RefCell::new(window)), Rc::new(RefCell::new(state))))
}

/// After calling prepare() call new_window() for each extra window.
pub async fn new_window(config: InitialConfiguration, event_loop: &EventLoop<ControllerEvent>) -> Result<(Rc<RefCell<GpuState>>, Rc<RefCell<Window>>, Rc<RefCell<TeState>>), Error> {
    let img = match ImageReader::open(&config.icon_path)?.decode() {
        Ok(img) => img.to_rgba8(),
        Err(_) => panic!("Couldn't find icon"),
    };

    let wb = WindowBuilder::new()
        .with_title(&config.window_name)
        .with_inner_size(dpi::LogicalSize::new(config.screen_width, config.screen_height))
        .with_window_icon(Some(match Icon::from_rgba(img.into_raw(), 64, 64) {
            Ok(icon) => icon,
            Err(_) => panic!("Couldn't get raw data")
        }));

    let window = wb.build(event_loop).expect("Unable to create window");

    let gpu = GpuState::new(window.inner_size(), &window).await;
    let state = TeState::new(&window, &gpu, config).await;

    Ok((Rc::new(RefCell::new(gpu)), Rc::new(RefCell::new(window)), Rc::new(RefCell::new(state))))
}
