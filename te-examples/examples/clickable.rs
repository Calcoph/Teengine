use std::rc::Rc;

use te_player::{event_loop::{Event, TextSender}, te_winit::{event_loop::ControlFlow, event::{WindowEvent, ElementState}, dpi::PhysicalSize}};
use te_renderer::{model::{Model, ModelVertex}, state::{TeColor, TeState, GpuState, Section, Text}, text::FontReference};
use wgpu::{ImageDataLayout, Extent3d};

pub(crate) const SQUARE_VERT: &[ModelVertex] = &[
    ModelVertex {
        position: [-0.025, 0.0, -0.025],
        tex_coords: [0.0, 0.0],
    },
    ModelVertex {
        position: [-0.025, 0.0, 0.025],
        tex_coords: [0.0, 1.0],
    },
    ModelVertex {
        position: [0.025, 0.0, -0.025],
        tex_coords: [1.0, 0.0],
    },
    ModelVertex {
        position: [0.025, 0.0, 0.025],
        tex_coords: [0.0, 0.0],
    }
];

pub(crate) const SQUARE_IND: &[u32] = &[
    1, 3, 2,
    1, 2, 0
];

const WIN_WIDTH: u32 = 512;
const WIN_HEIGHT: u32 = 512;
pub(crate) fn main() {
    let config = te_renderer::initial_config::InitialConfiguration {
        screen_width: WIN_WIDTH,
        screen_height: WIN_HEIGHT,
        camera_pitch: -90.0,
        camera_position: (0.0, 1.1, 0.0),
        camera_sensitivity: 10.0,
        camera_speed: 2.0,
        ..Default::default()
    };
    let (
        event_loop,
        te_gpu,
        te_window,
        te_state
    ) = pollster::block_on(te_player::prepare(config, true)).expect("Couldn't initilize");
    let mut te_gpu = Rc::<_>::try_unwrap(te_gpu).expect("Unreachable").into_inner();
    let te_window = Rc::<_>::try_unwrap(te_window).expect("Unreachable").into_inner();
    let mut te_state = Rc::<_>::try_unwrap(te_state).expect("Unreachable").into_inner();

    let img = image::open("resources/default_texture.png").expect("make sure default_texture.png exists in \"resources\" directory");
    let img = img.as_rgba8().expect(r#""resources/default_texture.png" should have an alpha channel, but it doesn't"#);
    let square_model = Model::new_simple(SQUARE_VERT.into(), SQUARE_IND.into(), img, &te_gpu, &te_state.instances.layout);

    let x_pos = 100.0;
    let y_pos = 200.0;
    let mut click = None;
    let mut mouse_pos = (0, 0);
    te_state.place_custom_model("model_name", &te_gpu, ((x_pos/1000.0)-0.5, 0.0, (y_pos/1000.0)-0.5), Some(square_model)).expect("Unreachable");
    for j in 0..5 {
        for i in 0..5 {
            let posx = ((i as f32)/20.0)-0.3;
            let posy = ((j as f32)/20.0)-0.3;
            let square_model = Model::new_simple(SQUARE_VERT.into(), SQUARE_IND.into(), img, &te_gpu, &te_state.instances.layout);

            // This is bad practice. Since model_name{i}_{j} is the same model as model_name, they both should be model_name. and the last argument should be None instead of Some(square_model)
            te_state.place_custom_model(&format!("model_name{i}_{j}"), &te_gpu, (posx, -((i+j) as f32)/100.0, posy), Some(square_model)).expect("Unreachable");
        }
    }

    //let sprite = te_state.borrow_mut().place_sprite("default_texture.png", &te_gpu.as_ref().borrow(), Some((50.0, 50.0)), (x_pos, y_pos, 50.0));
    let mut last_render = std::time::Instant::now();
    /* 
    let (vertices, indices) = get_model();

    let img = image::open("resources/default_texture.png").unwrap();
    let img = img.as_rgba8().unwrap();

    let model = te_renderer::model::Model::new_simple(vertices, indices, img, &te_gpu.as_ref().borrow(), &te_state.as_ref().borrow().instances.layout);

    te_state.borrow_mut().place_custom_model("pepe", &te_gpu.as_ref().borrow(), (0.0, 0.0, 0.0), Some(model));
    */

    te_state.bgcolor = TeColor::new(0.5, 0.5, 0.5);
    let font = te_state.load_font(String::from("CascadiaCode.ttf"), &te_gpu).expect("Could not find font");
    let mut my_text = MyText::new(font);
    let mut see_clickable = false;
    let mut unpadded_width = WIN_WIDTH;
    let mut padded_width = pad(unpadded_width);
    te_window.set_inner_size(PhysicalSize{ width: WIN_WIDTH, height: WIN_HEIGHT }); // Because we are dealing always with PhysicalSize but te_player::prepare() uses LogicalSize
    let mut clickable_depth_texture = te_renderer::texture::Texture::create_depth_texture_surfaceless(&te_gpu.device, WIN_WIDTH, WIN_HEIGHT, "clickable_depth_texture");

    let mut last_render_time = std::time::Instant::now();
    event_loop.run(move |event, _window_target, control_flow| {
        *control_flow = ControlFlow::Poll;
        match &event {
            Event::WindowEvent { window_id, event } if *window_id == te_window.id() => {
                match event {
                    WindowEvent::Resized(size) => {
                        let size = if see_clickable {
                            let old_width = padded_width;
                            unpadded_width = size.width;
                            padded_width = pad(unpadded_width);
                            if old_width != padded_width { // adjust the window to the padded size
                                te_window.set_inner_size(PhysicalSize{ width: padded_width, height: te_state.size.height });
                            };
                            PhysicalSize { width: padded_width, height: size.height }
                        } else {
                            unpadded_width = size.width;
                            padded_width = pad(unpadded_width);
                            *size
                        };
                        te_gpu.resize(size);
                        te_state.resize(size);
                        te_state.camera.resize_2d_space(size.width, size.height, &te_gpu.queue);
                        clickable_depth_texture = te_renderer::texture::Texture::create_depth_texture_surfaceless(&te_gpu.device, padded_width, size.height, "clickable_depth_texture");
                    },
                    WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit, //control_flow is a pointer to the next action we wanna do. In this case, exit the program
                    WindowEvent::ScaleFactorChanged { scale_factor: _, new_inner_size } => {
                        te_gpu.resize(**new_inner_size);
                        te_state.resize(**new_inner_size)
                    },
                    WindowEvent::CursorMoved { device_id: _, position, .. } => {
                        mouse_pos = (position.x.floor() as u32, position.y.floor() as u32)
                    },
                    WindowEvent::KeyboardInput { device_id: _, input, is_synthetic: _ } => {
                        if let ElementState::Pressed = input.state {
                            match input.virtual_keycode {
                                Some(key) => match key {
                                    te_player::te_winit::event::VirtualKeyCode::M => {
                                        see_clickable = !see_clickable;
                                        let size = if see_clickable {
                                            PhysicalSize{ width: padded_width, height: te_state.size.height }
                                        } else {
                                            PhysicalSize{ width: unpadded_width, height: te_state.size.height }
                                        };
                                        te_window.set_inner_size(size);
                                    },
                                    _ => ()
                                },
                                None => (),
                            }
                        }
                    },
                    WindowEvent::MouseInput { device_id: _, state, button, .. } => {
                        match (state, button) {
                            (
                                te_player::te_winit::event::ElementState::Pressed,
                                te_player::te_winit::event::MouseButton::Left
                            ) => click = Some(mouse_pos),
                            _ => ()
                        }
                    }
                    _ => ()
                }
            },
            Event::Suspended => *control_flow = ControlFlow::Wait,
            Event::Resumed => (),
            Event::MainEventsCleared => {
                te_window.request_redraw()
            },
            Event::RedrawRequested(window_id) => if *window_id == te_window.id() {
                pollster::block_on(render(
                    &mut last_render,
                    &mut last_render_time,
                    &mut te_state,
                    &te_gpu,
                    click,
                    &mut my_text,
                    see_clickable,
                    padded_width,
                    &clickable_depth_texture
                ));
                click = None;
            },
            _ => ()
        }
    });
}

async fn render(
    last_render: &mut std::time::Instant,
    last_render_time: &mut std::time::Instant,
    te_state: &mut TeState, te_gpu: &GpuState,
    click: Option<(u32, u32)>,
    texts: &mut MyText,
    see_clickable: bool,
    padded_width: u32,
    clickable_depth_texture: &te_renderer::texture::Texture
) {
    /* let t_diff = SPEED * dt.as_secs_f32();
    
    path.advance(t_diff);

    let pos = path.get_pos();
    x_pos = (pos.x/1000.0)-0.5;
    y_pos = (pos.y/1000.0)-0.5;

    //te_s.set_instance_position(&sprite, (x_pos, y_pos, 50.0), &te_g.queue);
    te_state.set_instance_position(&square, (x_pos, 0.0, y_pos), &te_gpu.queue);
    te_state.camera.move_absolute((x_pos, 2.0, y_pos)); */
    
    // BOILERPLATE
    let now = std::time::Instant::now();
    let dt = now - *last_render_time;
    *last_render = now;
    *last_render_time = now;
    te_state.update(dt, &te_gpu);
    let output = te_gpu.surface.get_current_texture().expect("Couldn't get surface texture");
    let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
    let mut encoders = te_renderer::state::TeState::prepare_render(&te_gpu);
    //te_state.render(&view, &te_gpu, &mut encoder, &[]);

    let click_texture = te_gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: Extent3d { width: padded_width, height: te_state.size.height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Uint,
        usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[wgpu::TextureFormat::R32Uint]
    });
    if see_clickable {
        let clickable_encoder = &mut encoders[0];
        te_state.clicakble_mask(&view, &te_gpu, clickable_encoder, true, None);
    } else {
        texts.draw_text(|texts| {
            te_state.render(&view, &te_gpu, &mut encoders, texts)
        });
    }
    let clickable_encoder = &mut encoders[0];
    te_state.clicakble_mask(
        &click_texture.create_view(&wgpu::TextureViewDescriptor::default()),
        &te_gpu,
        clickable_encoder,
        false,
        Some(&clickable_depth_texture.view)
    );
    let size = padded_width * te_state.size.height * 4;
    let destination = te_gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: size as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let destination_buffer = wgpu::ImageCopyBufferBase { buffer: &destination, layout: ImageDataLayout {
        offset: 0,
        //bytes_per_row: NonZeroU32::new(te_state.size.width * 4),
        bytes_per_row: Some(padded_width * 4),
        rows_per_image: None
    }};
    clickable_encoder.copy_texture_to_buffer(
        click_texture.as_image_copy(),
        destination_buffer,
        wgpu::Extent3d { width: padded_width, height: te_state.size.height, depth_or_array_layers: 1 },
    );

    te_state.end_render(&te_gpu, encoders);

    let slice = destination.slice(..);
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).expect("Couldn't send"));

    te_gpu.device.poll(wgpu::Maintain::Wait);
    let result: Vec<u32> = match receiver.receive().await {
        Some(Ok(())) => {
            // Gets contents of buffer
            let data = slice.get_mapped_range();
            // Since contents are got in bytes, this converts these bytes back to u32
            let result = bytemuck::cast_slice(&data).to_vec();

            // With the current interface, we have to make sure all mapped views are
            // dropped before we unmap the buffer.
            drop(data);
            destination.unmap(); // Unmaps buffer from memory
                                    // If you are familiar with C++ these 2 lines can be thought of similarly to:
                                    //   delete myPointer;
                                    //   myPointer = NULL;
                                    // It effectively frees the memory

            // Returns data from buffer
            result
        },
        _ => panic!("AAA")
    };

    if let Some((x, y)) = click {
        let num = result[(y*te_state.size.width + x) as usize];
        println!("Clicked on square {num}!");
        match te_state.find_clicked_instance(num) {
            Some(instance) => {
                let name = instance.get_name();
                println!("That is {name}")
            },
            None => println!("You clicked nowhere"),
        }
    }

    output.present();
    te_state.text.after_present();
}


struct MyText {
    sections: Vec<(FontReference, Vec<Section<'static>>)>
}

impl MyText {
    fn new(font: FontReference) -> MyText {
        MyText {
            sections: vec![(font, vec![
                Section {
                    screen_position: (30.0, 350.0),
                    bounds: (WIN_WIDTH as f32 - 30.0, WIN_HEIGHT as f32),
                    text: vec![
                        Text::new("Click on any square and see how it prints a different value for each one. Press M to toggle between this view and view showing clickable elements.")
                            .with_color([1.0, 1.0, 1.0, 1.0])
                            .with_scale(30.0)
                    ],
                    ..Section::default()
                }
            ])]
        }
    }
}

impl TextSender for MyText {
    fn draw_text<T: FnMut(&[(FontReference, Vec<Section>)])>(&mut self, mut drawer: T ) {
        drawer(&self.sections)
    }
}

fn pad(width: u32) -> u32 {
    // 4 bytes per poxel
    let dif = (width*4) % wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;

    if dif != 0 {
        width + (wgpu::COPY_BYTES_PER_ROW_ALIGNMENT - dif)/4
    } else {
        width
    }
}