use std::{rc::Rc, cell::RefCell};

use te_player::{imgui::{PrepareResult, ImguiState}, event_loop::PlaceholderTextSender};
use te_renderer::{initial_config::InitialConfiguration, state::TeState};

fn main() {
    pollster::block_on(as_main());
}

async fn as_main() {
    let PrepareResult {
        event_loop,
        gpu,
        window,
        te_state,
        context,
        platform,
        renderer,
    } = te_player::imgui::prepare(InitialConfiguration {
        screen_width: 1000,
        screen_height: 500,
        ..InitialConfiguration::default()
    }, true).await.expect("Failed init");

    let my_imgui = MyImgui { te_state: te_state.clone() };
    te_player::imgui::event_loop::run(event_loop, window, gpu, te_state, my_imgui, platform, context, renderer, PlaceholderTextSender::new(), Box::new(|_, _, _| {}));
}

struct MyImgui {
    te_state: Rc<RefCell<TeState>>,
}

impl ImguiState for MyImgui {
    fn create_ui(&mut self, ui: &te_player::imgui::Ui) {
        ui.window("Imgui example")
            .build(|| {
                let bgcolor = &mut self.te_state.borrow_mut().bgcolor;
                let mut red = bgcolor.get_red();
                let mut green = bgcolor.get_green();
                let mut blue = bgcolor.get_blue();

                ui.slider("Red", 0.0, 1.0, &mut red);
                ui.slider("Green", 0.0, 1.0, &mut green);
                ui.slider("Blue", 0.0, 1.0, &mut blue);

                bgcolor.set_red(red);
                bgcolor.set_green(green);
                bgcolor.set_blue(blue);
            });
    }
}
