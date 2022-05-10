extern crate sdl2;

use sdl2::pixels::Color;
use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use std::time::Duration;

use super::mnist;

pub fn show_mnist_image(image: &mnist::Image) {
    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();

    let window = video_subsystem
        .window("MNIST", 28 * 10, 28 * 10)
        .position_centered()
        .build()
        .unwrap();

    let mut canvas = window.into_canvas().build().unwrap();
    canvas.set_draw_color(Color::RGB(0, 0, 0));
    canvas.clear();

    for (i, pixel) in image.pixels.iter().enumerate() {
        let x = i % 28;
        let y = i / 28;

        let color = Color::RGB(*pixel, 0, 0);

        canvas.set_draw_color(color);
        canvas.fill_rect(sdl2::rect::Rect::new(x as i32 * 10, y as i32 * 10, 10, 10)).unwrap();
    }

    canvas.present();

    let mut event_pump = sdl_context.event_pump().unwrap();
    let mut running = true;

    while running {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit {..} |
                Event::KeyDown { keycode: Some(Keycode::Escape), .. } => {
                    running = false;
                },
                _ => {}
            }
        }

        ::std::thread::sleep(Duration::new(0, 1_000_000_000u32 / 60));
    }
}
