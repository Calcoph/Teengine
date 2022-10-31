use std::{cell::RefCell, rc::Rc};

use te_player::{event_loop::{Event, TextSender}, te_winit::event::{WindowEvent, StartCause}};
use te_renderer::{initial_config::InitialConfiguration, state::{Section, Text}, text::FontReference};

fn main() {
    pollster::block_on(as_main());
}

const WIN_WIDTH: u32 = 1000;
const WIN_HEIGHT: u32 = 500;

async fn as_main() {
    let (
        event_loop,
        gpu,
        window,
        te_state,
    ) = te_player::prepare(InitialConfiguration {
        screen_width: WIN_WIDTH,
        screen_height: WIN_HEIGHT,
        ..InitialConfiguration::default()
    }, true).await.expect("Failed init");

    let font = {
        let te_s = &mut te_state.borrow_mut();
        let gpu_ = &gpu.as_ref().borrow();
        te_s.load_font(String::from("CascadiaCode.ttf"), gpu_).expect("Could not find font")
    };

    let text = Rc::new(RefCell::new(MyText::new(font)));
    te_player::event_loop::run(event_loop, window.clone(), gpu, te_state, text.clone(), Box::new(move |event| {
        match event {
            Event::NewEvents(cause) => match cause {
                StartCause::Init => {
                    
                },
                _ => ()
            },
            Event::WindowEvent { window_id, event } if window_id == window.borrow().id() => match event {
                WindowEvent::Resized(s) => text.borrow_mut().resize(s.width as f32, s.height as f32),
                _ => ()
            },
            _ => ()
        }
    }));
}

struct MyText {
    sections: Vec<(FontReference, Vec<Section<'static>>)>
}

impl MyText {
    fn new(font: FontReference) -> MyText {
        MyText {
            sections: vec![(font, vec![
                Section {
                    screen_position: (30.0, 30.0),
                    bounds: (WIN_WIDTH as f32 - 30.0, WIN_HEIGHT as f32),
                    text: vec![
                        Text::new("hello").with_color([1.0, 1.0, 1.0, 1.0])
                    ],
                    ..Section::default()
                },
                Section {
                    screen_position: (30.0, 60.0),
                    bounds: (WIN_WIDTH as f32 - 30.0, WIN_HEIGHT as f32),
                    text: vec![
                        Text::new("hello, but red").with_color([1.0, 0.0, 0.0, 1.0])
                    ],
                    ..Section::default()
                },
                Section {
                    screen_position: (30.0, 90.0),
                    bounds: (WIN_WIDTH as f32 - 30.0, WIN_HEIGHT as f32),
                    text: vec![
                        Text::new("HELLO")
                            .with_color([1.0, 1.0, 1.0, 1.0])
                            .with_scale(80.0)
                    ],
                    ..Section::default()
                },
                Section {
                    screen_position: (30.0, 300.0),
                    bounds: (WIN_WIDTH as f32 - 30.0, WIN_HEIGHT as f32),
                    text: vec![
                        Text::new("Resize my window horizontally and you will see that the text wraps, this is done with the `bounds` property of the Section struct")
                            .with_color([1.0, 1.0, 1.0, 1.0])
                            .with_scale(30.0)
                    ],
                    ..Section::default()
                }
            ])]
        }
    }

    fn resize(&mut self, width: f32, height: f32) {
        for (_, i) in self.sections.iter_mut() {
            for j in i {
                j.bounds = (width-30.0, height);
            }
        }
    }
}

impl TextSender for MyText {
    fn draw_text<T: FnMut(&[(FontReference, Vec<Section>)])>(&mut self, mut drawer: T ) {
        drawer(&self.sections)
    }
}
