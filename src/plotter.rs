use std::collections::{
    HashMap,
    HashSet,
};

use sdl2::{
    pixels::Color,
    event::Event,
    keyboard,
    rect::{
        Point
    }
};

use crossbeam_channel::{
    Receiver,
};

pub trait DataPoint: Send + Copy + std::fmt::Debug {
    fn x(&self) -> f32;
    fn y(&self) -> f32;
    fn series_name(&self) -> &str;
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.series_name())
    }
}

pub trait Series<P> where
    P: DataPoint,
{
    fn name(&self) -> &str;
    fn data(&self) -> &[P];

    fn min_x(&self) -> f32;
    fn max_x(&self) -> f32;
    fn range_x(&self) -> f32 {
        self.max_x() - self.min_x()
    }

    fn min_y(&self) -> f32;
    fn max_y(&self) -> f32;
    fn range_y(&self) -> f32 {
        self.max_y() - self.min_y()
    }

    fn width(&self) -> f32;
    fn height(&self) -> f32;

    fn add(&mut self, point: P) -> bool;
    fn points(&mut self) -> &[Point];
}

pub struct SeriesData<P: DataPoint> {
    name: String,
    data: Vec<P>,
    points: Vec<Point>,
    points_need_update: bool,
    min_x: Option<f32>,
    max_x: Option<f32>,
    min_y: Option<f32>,
    max_y: Option<f32>,
    x_start: f32,
    x_end: f32,
    y_start: f32,
    y_end: f32,
}

impl <P> Series<P> for SeriesData<P> where P: DataPoint {
    fn name(&self) -> &str {
        &self.name
    }

    fn data(&self) -> &[P] {
        &self.data
    }

    fn add(&mut self, point: P) -> bool {
        if let Some(min_x) = self.min_x {
            if point.x() < min_x {
                self.min_x = Some(point.x());
            }
        } else {
            self.min_x = Some(point.x());
        }

        if let Some(max_x) = self.max_x {
            if point.x() > max_x {
                self.max_x = Some(point.x());
            }
        } else {
            self.max_x = Some(point.x());
        }

        if let Some(min_y) = self.min_y {
            if point.y() < min_y {
                self.min_y = Some(point.y());
            }
        } else {
            self.min_y = Some(point.y());
        }

        if let Some(max_y) = self.max_y {
            if point.y() > max_y {
                self.max_y = Some(point.y());
            }
        } else {
            self.max_y = Some(point.y());
        }

        self.data.push(point);

        if self.width() > 0.0 && self.height() > 0.0 {
            self.points_need_update = true;
            true
        } else {
            false
        }
    }

    fn min_x(&self) -> f32 {
        self.min_x.unwrap_or(0.0)
    }

    fn max_x(&self) -> f32 {
        self.max_x.unwrap_or(0.0)
    }

    fn min_y(&self) -> f32 {
        self.min_y.unwrap_or(0.0)
    }

    fn max_y(&self) -> f32 {
        self.max_y.unwrap_or(0.0)
    }

    fn points(&mut self) -> &[Point] {
        if self.points_need_update {
            if self.data.len() > self.width() as usize * 4 {
                let data = self.data.clone();
                self.data.clear();
                let factor = (data.len() as f32 / self.width() as f32) as usize;
                for (i, point) in data.iter().enumerate() {
                    if i % factor == 0 {
                        self.data.push(*point);
                    }
                }
            }

            self.points = self.data()
                .iter()
                .map(|p| {
                    let x = (p.x() - self.min_x()) / self.range_x();
                    let y = (p.y() - self.min_y()) / self.range_y();
                    Point::new(
                        ((x * self.width()) + self.x_start) as i32,
                        (self.height() - (y * self.height()) + self.y_start) as i32,
                    )
                })
                .collect();
            self.points_need_update = false;
        }


        &self.points
    }

    fn width(&self) -> f32 {
        self.x_end - self.x_start
    }

    fn height(&self) -> f32 {
        self.y_end - self.y_start
    }
}

pub struct SeriesCollection<P: DataPoint> {
    names: HashSet<std::string::String>,
    series: HashMap<std::string::String, SeriesData<P>>,
    x_start: f32,
    x_end: f32,
    y_start: f32,
    y_end: f32,
}

impl <P: DataPoint> SeriesCollection<P> {
    fn new(x_start: f32, x_end: f32, y_start: f32, y_end: f32) -> Self {
        Self {
            names: HashSet::new(),
            series: HashMap::new(),
            x_start,
            x_end,
            y_start,
            y_end,
        }
    }

    fn add(&mut self, point: P) -> bool {
        let series_name = point.series_name();

        self.names.insert(series_name.to_string());

        let (x_start, x_end, y_start, y_end) = (
            self.x_start,
            self.x_end,
            self.y_start,
            self.y_end,
        );

        let series = self.series.entry(series_name.to_string()).or_insert_with(|| {
            SeriesData {
                name: series_name.to_string(),
                data: Vec::new(),
                points: Vec::new(),
                points_need_update: false,
                min_x: None,
                max_x: None,
                min_y: None,
                max_y: None,
                x_start, x_end,
                y_start, y_end,
            }
        });

        series.add(point)
    }

    fn for_each_series<F>(&mut self, mut f: F) where
        F: FnMut(&mut SeriesData<P>)
    {
        for series in self.series.values_mut() {
            f(series);
        }
    }
}

pub fn plot<P>(receiver: &mut Receiver<P>) where
    P: DataPoint,
{
    let sdl_context = sdl2::init().expect("SDL2 context initialization");
    let video_subsystem = sdl_context.video().expect("SDL2 video initialization");

    let width = 800;
    let height = 600;

    let window = video_subsystem
        .window("ML Training", width, height)
        .position_centered()
        .build()
        .expect("SDL2 window initialization");

    let mut event_pump = sdl_context.event_pump().expect("SDL2 event pump initialization");

    let mut canvas = window.into_canvas().build().expect("SDL2 canvas initialization");

    let mut series_collection = SeriesCollection::new(
        0.0, width as f32,
        0.0, height as f32,
    );

    'window: loop {
        let mut need_update = false;

        match receiver.try_recv() {
            Ok(data_point) => {
                need_update = series_collection.add(data_point);
            },
            Err(_) => {},
        }

        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. } => break 'window,
                Event::KeyDown { keycode: Some(keycode), .. } => {
                    match keycode {
                        keyboard::Keycode::Escape => break 'window,
                        _ => {},
                    }
                },
                _ => {},
            }
        }

        if need_update {
            canvas.set_draw_color(Color::RGB(255, 255, 255));
            canvas.clear();

            canvas.set_draw_color(Color::RGB(0, 0, 0));

            series_collection.for_each_series(|s| {
                canvas.draw_lines(s.points()).expect("SDL2 draw lines");
            });

            canvas.present();
        }
    }
}
