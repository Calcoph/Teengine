use std::{cell::RefCell, rc::Rc};

use te_player::{
    event_loop::PlaceholderTextSender,
    imgui::{ImguiState, PrepareResult},
};
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
        gamepad_handler
    } = te_player::imgui::prepare(
        InitialConfiguration {
            screen_width: 1000,
            screen_height: 500,
            ..InitialConfiguration::default()
        },
        true,
        true,
        true
    )
    .await
    .expect("Failed init");

    let my_imgui = MyImgui {
        te_state: te_state.clone(),
        opened: true
    };
    te_player::imgui::event_loop::run(
        event_loop,
        window,
        gpu,
        te_state,
        my_imgui,
        platform,
        context,
        renderer,
        PlaceholderTextSender::new(),
        Box::new(|_, _, _| {}),
        gamepad_handler
    );
}

struct MyImgui {
    te_state: Rc<RefCell<TeState>>,
    opened: bool
}

impl ImguiState for MyImgui {
    fn create_ui(&mut self, ui: &te_player::imgui::Ui) {
        ui.window("Hello world")
            .size([300.0, 110.0], te_player::imgui::Condition::FirstUseEver)
            .build(|| {
                ui.text_wrapped("Hello world!");
                ui.text_wrapped("こんにちは世界！");
                if ui.button("ads") {
                }

                ui.button("This...is...imgui-rs!");
                ui.separator();
                let mouse_pos = ui.io().mouse_pos;
                ui.text(format!(
                    "Mouse Position: ({:.1},{:.1})",
                    mouse_pos[0], mouse_pos[1]
                ));
            });
        
        ui.window("Hello world2")
            .size([300.0, 110.0], te_player::imgui::Condition::FirstUseEver)
            .build(|| {
                ui.text_wrapped("Hello world!");
                ui.text_wrapped("こんにちは世界！");
                if ui.button("ads") {
                }

                ui.button("This...is...imgui-rs!");
                ui.separator();
                let mouse_pos = ui.io().mouse_pos;
                ui.text(format!(
                    "Mouse Position: ({:.1},{:.1})",
                    mouse_pos[0], mouse_pos[1]
                ));
            });
        /* ui.window("Imgui example").build(|| {
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

        ui.show_demo_window(&mut self.opened) */
    }
}
