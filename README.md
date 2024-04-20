Graphics engine composed of the following crates:

* **te-renderer**: The core of the project, the rendering engine. Supports view frustrum culling, 2D sprites, 3D models, basic animation, among other things.
* **te-player**: Displays windows and handles winit events to talk to the renderer. Supports: easy imgui integration.
* **te-gamepad**: Glue for winit+gilrs to also emit gamepad events.
* **te-mapmaker**: GUI Map editor using te-renderer adn imgui.
* **te-examples**: Various examples that show how to use the project.

# WIP

Lacks documentation and is made by someone learning about computer graphics. Use at own risk.

It has some exapmles in the te-examples directory
 * `cargo run --example empty_window.rs`
 * `cargo run --example multiple_window.rs`
 * `cargo run --example text.rs`
 * `cargo run --example imgui_example.rs`

# Projects that use Teengine
* [TeProcedural](https://github.com/Calcoph/TeProcedural)
