pub mod event_loop;

use std::{io::Error, cell::RefCell, rc::Rc};
use te_renderer::{initial_config::InitialConfiguration, state::{State, GpuState}};
use winit::{window::{WindowBuilder, Icon, Window}, dpi, event_loop::EventLoop};
use image::io::Reader as ImageReader;

use te_gamepad::gamepad::{self, ControllerEvent};

pub async fn prepare(config: InitialConfiguration) -> Result<(EventLoop<ControllerEvent>, Rc<RefCell<GpuState>>, Window, Rc<RefCell<State>>), Error> {
    env_logger::init();
    let img = match ImageReader::open("icon.png")?.decode() {
        Ok(img) => img.to_rgba8(),
        Err(_) => panic!("Couldn't find icon"),
    };
    let event_loop = EventLoop::with_user_event();
    gamepad::listen(event_loop.create_proxy());
    let wb = WindowBuilder::new()
        .with_title("Tilengine")
        .with_inner_size(dpi::LogicalSize::new(config.screen_width, config.screen_height))
        .with_window_icon(Some(match Icon::from_rgba(img.into_raw(), 64, 64) {
            Ok(icon) => icon,
            Err(_) => panic!("Couldn't get raw data")
        }));

    let window = wb.build(&event_loop)
        .unwrap();

    let gpu = GpuState::new(window.inner_size(), &window).await;
    let state = State::new(&window, &gpu, config).await;

    Ok((event_loop, Rc::new(RefCell::new(gpu)), window, Rc::new(RefCell::new(state))))
}