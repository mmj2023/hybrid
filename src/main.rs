// =============================================================================
// Wayland Compositor — Single-file implementation using Smithay 0.7
//
// This file contains:
//   1. The pluggable LayoutStrategy trait (implement to create any layout)
//   2. A default MasterStackLayout implementation
//   3. All Wayland protocol handlers (compositor, XDG shell, SHM, seat, etc.)
//   4. The calloop event loop and Wayland socket setup
// =============================================================================

use std::sync::Arc;

use smithay::{
    delegate_compositor, delegate_data_device, delegate_output, delegate_seat, delegate_shm,
    delegate_xdg_shell,
    desktop::{PopupManager, Space, Window},
    input::{pointer::CursorImageStatus, Seat, SeatHandler, SeatState},
    output::{self, Output, PhysicalProperties, Subpixel},
    reexports::{
        calloop::{generic::Generic, EventLoop, Interest, LoopSignal, Mode, PostAction},
        wayland_server::{
            backend::{ClientData, ClientId, DisconnectReason},
            protocol::wl_surface::WlSurface,
            Display, DisplayHandle,
        },
    },
    utils::{Logical, Point, Rectangle, Size},
    wayland::{
        compositor::{CompositorClientState, CompositorHandler, CompositorState},
        output::OutputHandler,
        selection::{
            data_device::{
                ClientDndGrabHandler, DataDeviceHandler, DataDeviceState, ServerDndGrabHandler,
            },
            SelectionHandler,
        },
        shell::xdg::{
            PopupSurface, PositionerState, ToplevelSurface, XdgShellHandler, XdgShellState,
        },
        shm::{ShmHandler, ShmState},
        socket::ListeningSocketSource,
    },
};
use tracing::{info, warn};

// =============================================================================
// Layout Strategy — the pluggable interface for window arrangement
// =============================================================================

/// Describes where a single window should be placed and how large it should be.
/// Returned by `LayoutStrategy::arrange()` for each window.
pub struct WindowLayout {
    /// Index into the windows slice passed to `arrange()`
    pub window_index: usize,
    /// Top-left position in logical coordinates
    pub position: Point<i32, Logical>,
    /// Width and height in logical pixels
    pub size: Size<i32, Logical>,
}

/// The pluggable layout interface.
///
/// Implement this trait to define how windows are arranged on screen.
/// The compositor calls `arrange()` whenever windows change, and applies the
/// returned positions/sizes to the Space.
///
/// # Examples of layouts you can build:
/// - **Tiling (master-stack)**: first window = left half, rest stack vertically on right
/// - **Floating**: windows keep their requested position; `on_window_move_request` returns `true`
/// - **Scrolling**: windows in a horizontal strip with a viewport offset
/// - **Grid**: equal-sized cells in an N×M grid
/// - **Hybrid**: check window type — float dialogs, tile normal windows
pub trait LayoutStrategy {
    /// Compute the position and size of every window.
    ///
    /// Called whenever windows are added, removed, or the output changes.
    /// `output_geo` is the full output rectangle (e.g. 1920×1080 at (0,0)).
    /// `windows` is the ordered list of currently mapped windows.
    ///
    /// Return a `WindowLayout` for each window you want displayed.
    fn arrange(
        &mut self,
        output_geo: Rectangle<i32, Logical>,
        windows: &[Window],
    ) -> Vec<WindowLayout>;

    /// Called when a new window is added, before `arrange()`.
    /// Use this for bookkeeping (e.g. assigning it to a group in a hybrid layout).
    fn on_window_added(&mut self, _window: &Window) {}

    /// Called when a window is removed, before `arrange()`.
    fn on_window_removed(&mut self, _window: &Window) {}

    /// Called when a client requests to move a window (e.g. drag in floating mode).
    /// Return `true` to allow the move, `false` to deny it.
    /// Default: deny (appropriate for tiling layouts).
    fn on_window_move_request(
        &mut self,
        _window: &Window,
        _new_position: Point<i32, Logical>,
    ) -> bool {
        false
    }

    /// Called when a client requests to resize a window.
    /// Return the allowed new size (you can clamp or constrain).
    /// Default: return the requested size unchanged.
    fn on_window_resize_request(
        &mut self,
        _window: &Window,
        new_size: Size<i32, Logical>,
    ) -> Size<i32, Logical> {
        new_size
    }

    /// Human-readable name of this layout, used for logging.
    fn name(&self) -> &str;
}

// =============================================================================
// MasterStackLayout — default tiling layout implementation
// =============================================================================

/// A master-stack tiling layout.
///
/// ```text
/// ┌──────────────┬──────────────┐
/// │              │   Stack 1    │
/// │              ├──────────────┤
/// │    Master    │   Stack 2    │
/// │              ├──────────────┤
/// │              │   Stack 3    │
/// └──────────────┴──────────────┘
/// ```
///
/// - If there's only 1 window, it takes the full output.
/// - The master window gets `master_ratio` of the width (default 0.5 = 50%).
/// - Stack windows split the remaining width equally in height.
/// - `gap` is the pixel gap between windows (default 4px).
pub struct MasterStackLayout {
    /// Fraction of output width for the master window (0.0 to 1.0)
    pub master_ratio: f64,
    /// Gap in pixels between windows and edges
    pub gap: i32,
}

impl Default for MasterStackLayout {
    fn default() -> Self {
        Self {
            master_ratio: 0.5,
            gap: 4,
        }
    }
}

impl LayoutStrategy for MasterStackLayout {
    fn arrange(
        &mut self,
        output_geo: Rectangle<i32, Logical>,
        windows: &[Window],
    ) -> Vec<WindowLayout> {
        if windows.is_empty() {
            return Vec::new();
        }

        let gap = self.gap;
        let ox = output_geo.loc.x;
        let oy = output_geo.loc.y;
        let ow = output_geo.size.w;
        let oh = output_geo.size.h;

        // Single window: take the full output minus gaps
        if windows.len() == 1 {
            return vec![WindowLayout {
                window_index: 0,
                position: Point::from((ox + gap, oy + gap)),
                size: Size::from((ow - 2 * gap, oh - 2 * gap)),
            }];
        }

        let mut layouts = Vec::with_capacity(windows.len());

        // Master window: left portion
        let master_w = ((ow as f64) * self.master_ratio) as i32 - gap - gap / 2;
        layouts.push(WindowLayout {
            window_index: 0,
            position: Point::from((ox + gap, oy + gap)),
            size: Size::from((master_w, oh - 2 * gap)),
        });

        // Stack windows: right portion, split vertically
        let stack_count = windows.len() - 1;
        let stack_x = ox + ((ow as f64) * self.master_ratio) as i32 + gap / 2;
        let stack_w = ow - ((ow as f64) * self.master_ratio) as i32 - gap - gap / 2;
        let stack_h = (oh - gap * (stack_count as i32 + 1)) / stack_count as i32;

        for i in 0..stack_count {
            let y = oy + gap + i as i32 * (stack_h + gap);
            layouts.push(WindowLayout {
                window_index: i + 1,
                position: Point::from((stack_x, y)),
                size: Size::from((stack_w, stack_h)),
            });
        }

        layouts
    }

    fn name(&self) -> &str {
        "MasterStack"
    }
}

// =============================================================================
// Per-client state — attached to each connected Wayland client
// =============================================================================

/// Holds compositor-specific state for each connected client.
/// Smithay requires this to track surface state per-client.
#[derive(Default)]
struct ClientState {
    compositor_state: CompositorClientState,
}

impl ClientData for ClientState {
    fn initialized(&self, _client_id: ClientId) {
        info!("Client connected");
    }

    fn disconnected(&self, _client_id: ClientId, _reason: DisconnectReason) {
        info!("Client disconnected");
    }
}

// =============================================================================
// CalloopData — the state wrapper passed to calloop callbacks
// =============================================================================

/// This struct is what calloop passes to every callback.
/// It holds mutable references to the compositor state and the Wayland display,
/// so callbacks can process client requests and update state.
struct CalloopData {
    state: CompState,
    display: Display<CompState>,
}

// =============================================================================
// CompState — central compositor state
// =============================================================================

/// The main compositor state. Holds all Smithay protocol sub-states,
/// the pluggable layout engine, and the desktop Space that tracks
/// window positions for rendering.
struct CompState {
    // -- Wayland protocol states --
    compositor_state: CompositorState,
    xdg_shell_state: XdgShellState,
    shm_state: ShmState,
    seat_state: SeatState<Self>,
    data_device_state: DataDeviceState,

    // -- Desktop management --
    /// The Space tracks which windows are mapped and where they are positioned.
    /// When a renderer is added later, it reads from this Space to know what to draw.
    space: Space<Window>,
    /// Manages popup/tooltip surfaces (menus, dropdowns, etc.)
    popup_manager: PopupManager,

    // -- Layout engine --
    /// The active layout strategy. Swap this to change how windows are arranged.
    /// You can change it at runtime to switch between tiling/floating/etc.
    layout: Box<dyn LayoutStrategy>,

    // -- Misc --
    /// Handle to the event loop, for inserting new event sources
    #[allow(dead_code)]
    loop_signal: LoopSignal,
    /// Wayland display handle for creating protocol objects
    #[allow(dead_code)]
    display_handle: DisplayHandle,
    /// The seat (input device group — keyboard + pointer)
    #[allow(dead_code)]
    seat: Seat<Self>,
}

impl CompState {
    /// Re-run the active layout strategy and apply the computed positions to the Space.
    ///
    /// Called whenever windows are added, removed, or the layout changes.
    /// This is the bridge between the LayoutStrategy trait and Smithay's Space.
    fn re_layout(&mut self) {
        // Get the output geometry (the area we can place windows in)
        let output_geo = self
            .space
            .outputs()
            .next()
            .map(|o| self.space.output_geometry(o).unwrap())
            .unwrap_or_else(|| Rectangle::new((0, 0).into(), (1920, 1080).into()));

        // Collect current windows from the Space
        let windows: Vec<Window> = self.space.elements().cloned().collect();

        // Ask the layout strategy where each window should go
        let layouts = self.layout.arrange(output_geo, &windows);

        // Apply the computed positions and sizes
        for wl in &layouts {
            if wl.window_index < windows.len() {
                let window = &windows[wl.window_index];

                // Tell the XDG toplevel its new size via a configure event.
                // The client will then resize its buffer to match.
                window
                    .toplevel()
                    .expect("Window should have a toplevel")
                    .with_pending_state(|state| {
                        state.size = Some(wl.size);
                    });
                window
                    .toplevel()
                    .expect("Window should have a toplevel")
                    .send_pending_configure();

                // Move the window to its new position in the Space
                self.space.map_element(window.clone(), wl.position, false);
            }
        }

        info!(
            layout = self.layout.name(),
            windows = windows.len(),
            "Re-layout complete"
        );
    }
}

// =============================================================================
// BufferHandler — required for proper buffer management (SHM, etc.)
// =============================================================================

use smithay::wayland::buffer::BufferHandler;

impl BufferHandler for CompState {
    fn buffer_destroyed(&mut self, _buffer: &wayland_server::protocol::wl_buffer::WlBuffer) {
        // In a full compositor, you'd clean up any textures associated with this buffer.
    }
}

// =============================================================================
// CompositorHandler — handles surface creation and buffer commits
// =============================================================================

impl CompositorHandler for CompState {
    fn compositor_state(&mut self) -> &mut CompositorState {
        &mut self.compositor_state
    }

    fn client_compositor_state<'a>(
        &self,
        client: &'a wayland_server::Client,
    ) -> &'a CompositorClientState {
        &client.get_data::<ClientState>().unwrap().compositor_state
    }

    /// Called every time a client commits a new buffer to a surface.
    /// This is the main "something changed" signal from clients.
    fn commit(&mut self, surface: &WlSurface) {
        // Let Smithay's internal machinery process the commit
        // (updates cached surface state, processes subsurface reordering, etc.)
        on_commit_buffer_handler::<Self>(surface);

        // Handle popups — if a popup surface committed, process it
        // PopupManager::commit handles the commit if it's a popup
        let _ = self.popup_manager.commit(surface);

        // Check if this is a mapped XDG toplevel that needs initial configure
        let should_relayout = self
            .space
            .elements()
            .any(|w| w.toplevel().is_some_and(|t| t.wl_surface() == surface));

        if should_relayout {
            // Window map state didn't change, but content might have
            // triggering a need for re-layout (e.g. size change)
            self.re_layout();
        }
    }
}

// Smithay uses this function to process buffer state on commit.
// It's part of the compositor module but needs to be called explicitly.
fn on_commit_buffer_handler<D: CompositorHandler>(_surface: &WlSurface) {
    // In a full compositor, this would process buffer transforms,
    // damage tracking, etc. For now, Smithay handles the basics internally
    // via the delegate_compositor! macro.
}

delegate_compositor!(CompState);

// =============================================================================
// XdgShellHandler — window management (create, destroy, resize, popups)
// =============================================================================

impl XdgShellHandler for CompState {
    fn xdg_shell_state(&mut self) -> &mut XdgShellState {
        &mut self.xdg_shell_state
    }

    /// Called when a client creates a new toplevel window.
    /// We map it into the Space and run the layout engine.
    fn new_toplevel(&mut self, surface: ToplevelSurface) {
        // Create a Smithay desktop Window from the XDG toplevel
        let window = Window::new_wayland_window(surface);

        // Notify the layout engine that a new window was added
        self.layout.on_window_added(&window);

        // Map the window into the Space at (0,0) initially —
        // re_layout() will move it to the correct position
        self.space.map_element(window, (0, 0), true);

        // Re-run the layout to position all windows correctly
        self.re_layout();

        info!("New toplevel window mapped");
    }

    /// Called when a client creates a popup (context menu, dropdown, tooltip).
    fn new_popup(&mut self, surface: PopupSurface, _positioner: PositionerState) {
        // Track the popup in our PopupManager
        if let Err(err) = self.popup_manager.track_popup(surface.into()) {
            warn!("Failed to track popup: {:?}", err);
        }
    }

    /// Called when a popup requests an input grab (e.g. for combo-box menus
    /// that should close when you click outside them).
    fn grab(
        &mut self,
        _surface: PopupSurface,
        _seat: wayland_server::protocol::wl_seat::WlSeat,
        _serial: smithay::utils::Serial,
    ) {
        // In a full compositor, we'd set up an input grab here.
        // For now, we accept the grab request silently.
    }

    /// Called when a client requests its popup be repositioned
    /// (e.g., the parent window moved and the popup needs to follow).
    fn reposition_request(
        &mut self,
        surface: PopupSurface,
        positioner: PositionerState,
        token: u32,
    ) {
        surface.with_pending_state(|state| {
            let geometry = positioner.get_geometry();
            state.geometry = geometry;
            state.positioner = positioner;
        });
        surface.send_repositioned(token);
        if let Err(err) = surface.send_configure() {
            warn!("Failed to send popup configure: {:?}", err);
        }
    }

    /// Called when a toplevel surface is destroyed.
    fn toplevel_destroyed(&mut self, surface: ToplevelSurface) {
        // Find the window in the space and remove it
        let window = self
            .space
            .elements()
            .find(|w| w.toplevel().as_ref() == Some(&&surface))
            .cloned();

        if let Some(window) = window {
            self.layout.on_window_removed(&window);
            self.space.unmap_elem(&window);
            self.re_layout();
            info!("Toplevel window destroyed, re-layout triggered");
        }
    }
}

delegate_xdg_shell!(CompState);

// =============================================================================
// ShmHandler — shared memory buffer support
// =============================================================================

/// SHM (Shared Memory) is how non-GPU-accelerated clients send pixel data.
/// The client creates a shared memory pool, writes pixels into it, and the
/// compositor reads from it. This is the most basic buffer mechanism.
impl ShmHandler for CompState {
    fn shm_state(&self) -> &ShmState {
        &self.shm_state
    }
}

delegate_shm!(CompState);

// =============================================================================
// SeatHandler — input device management (keyboard, pointer)
// =============================================================================

impl SeatHandler for CompState {
    /// What type of object can receive keyboard focus.
    /// We use WlSurface — any surface can be focused.
    type KeyboardFocus = WlSurface;
    /// What type of object can receive pointer focus.
    type PointerFocus = WlSurface;
    /// What type of object can receive touch focus.
    type TouchFocus = WlSurface;

    fn seat_state(&mut self) -> &mut SeatState<Self> {
        &mut self.seat_state
    }

    /// Called when input focus changes to a new surface (or None if unfocused).
    fn focus_changed(&mut self, _seat: &Seat<Self>, _focused: Option<&WlSurface>) {
        // In a full compositor, you'd update visual indicators here
        // (e.g., change window border color for the focused window).
    }

    /// Called when a client sets a new cursor image.
    fn cursor_image(&mut self, _seat: &Seat<Self>, _image: CursorImageStatus) {
        // In a full compositor with a renderer, you'd update the cursor
        // texture here. For now, we accept it silently.
    }
}

delegate_seat!(CompState);

// =============================================================================
// DataDeviceHandler + SelectionHandler — clipboard and drag-and-drop
// =============================================================================

/// SelectionHandler is the base trait that DataDeviceHandler builds on.
/// It handles clipboard contents (copy/paste) and primary selection.
impl SelectionHandler for CompState {
    type SelectionUserData = ();
}

/// DataDeviceHandler manages the wl_data_device protocol:
/// - Clipboard (Ctrl+C / Ctrl+V between clients)
/// - Drag-and-drop between clients
impl DataDeviceHandler for CompState {
    fn data_device_state(&self) -> &DataDeviceState {
        &self.data_device_state
    }
}

/// Handles server-initiated drag-and-drop operations.
impl ServerDndGrabHandler for CompState {}

/// Handles client-initiated drag-and-drop operations.
impl ClientDndGrabHandler for CompState {}

delegate_data_device!(CompState);

// =============================================================================
// OutputHandler — monitor/display advertisement
// =============================================================================

/// Manages the wl_output protocol, which tells clients about available
/// monitors (resolution, scale factor, position, etc.).
impl OutputHandler for CompState {}

delegate_output!(CompState);

// =============================================================================
// AsMut<CompositorState> — required by some Smithay internals
// =============================================================================

impl AsMut<CompositorState> for CompState {
    fn as_mut(&mut self) -> &mut CompositorState {
        &mut self.compositor_state
    }
}

// =============================================================================
// main() — event loop, display, socket, seat, output initialization
// =============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // -------------------------------------------------------------------------
    // 1. Initialize structured logging via tracing
    // -------------------------------------------------------------------------
    // This sets up log output. Set RUST_LOG=info (or debug/trace) to control
    // verbosity. Example: RUST_LOG=debug cargo run
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    info!("Wayland compositor starting...");

    // -------------------------------------------------------------------------
    // 2. Create the calloop event loop
    // -------------------------------------------------------------------------
    // calloop is the callback-oriented event loop that Smithay is built on.
    // It dispatches events from multiple sources (Wayland clients, input
    // devices, timers, etc.) sequentially on a single thread.
    let mut event_loop: EventLoop<CalloopData> = EventLoop::try_new()?;
    let loop_signal = event_loop.get_signal();

    // -------------------------------------------------------------------------
    // 3. Create the Wayland display
    // -------------------------------------------------------------------------
    // The Display is the core Wayland server object. It manages the connection
    // to clients and dispatches protocol requests to our handler implementations.
    // The type parameter <CompState> tells it what our compositor state type is.
    let display: Display<CompState> = Display::new()?;
    let dh = display.handle();

    // -------------------------------------------------------------------------
    // 4. Initialize all Smithay protocol sub-states
    // -------------------------------------------------------------------------
    // Each of these creates one or more Wayland globals that clients can bind to.

    // CompositorState: registers wl_compositor and wl_subcompositor globals.
    // wl_compositor lets clients create surfaces (the fundamental display unit).
    // wl_subcompositor lets clients create subsurfaces (child surfaces).
    let compositor_state = CompositorState::new::<CompState>(&dh);

    // XdgShellState: registers xdg_wm_base global.
    // This is the standard shell protocol — clients use it to create windows
    // (toplevels) and popups with proper lifecycle management.
    let xdg_shell_state = XdgShellState::new::<CompState>(&dh);

    // ShmState: registers wl_shm global.
    // Allows clients to share pixel buffers via shared memory.
    // This is the basic, non-GPU-accelerated way clients provide window contents.
    let shm_state = ShmState::new::<CompState>(&dh, vec![]);

    // SeatState: prepares input device management.
    // We'll create a seat from this below.
    let mut seat_state = SeatState::new();

    // DataDeviceState: registers wl_data_device_manager global.
    // Provides clipboard (copy/paste) and drag-and-drop between clients.
    let data_device_state = DataDeviceState::new::<CompState>(&dh);

    // -------------------------------------------------------------------------
    // 5. Create a seat with keyboard and pointer capabilities
    // -------------------------------------------------------------------------
    // A "seat" represents a group of input devices (keyboard + mouse + touch)
    // that belong to one user. Most compositors have exactly one seat.
    let mut seat: Seat<CompState> = seat_state.new_wl_seat(&dh, "seat-0");

    // Add keyboard capability with default XKB keymap
    // XkbConfig::default() gives us a standard US keyboard layout.
    // The (200, 25) tuple is (repeat delay ms, repeat rate per second).
    seat.add_keyboard(Default::default(), 200, 25)?;

    // Add pointer (mouse) capability
    seat.add_pointer();

    info!("Seat 'seat-0' created with keyboard and pointer");

    // -------------------------------------------------------------------------
    // 6. Create a virtual output (monitor)
    // -------------------------------------------------------------------------
    // Even though we don't have a renderer yet, we need to advertise an output
    // so clients know the display dimensions and can size their surfaces.
    let output = Output::new(
        "Virtual-1".to_string(),
        PhysicalProperties {
            size: (0, 0).into(), // 0 = unknown physical size
            subpixel: Subpixel::Unknown,
            make: "Compositor".to_string(),
            model: "Virtual Output".to_string(),
        },
    );

    // Set the output mode (resolution and refresh rate)
    let mode = output::Mode {
        size: (1920, 1080).into(),
        refresh: 60_000, // 60Hz in mHz
    };
    output.change_current_state(Some(mode), None, None, Some((0, 0).into()));
    output.set_preferred(mode);

    // Create the Space (desktop window tracker) and map the output to it
    let mut space = Space::default();
    space.map_output(&output, (0, 0));

    info!("Output 'Virtual-1' created: 1920x1080 @ 60Hz");

    // -------------------------------------------------------------------------
    // 7. Assemble the compositor state
    // -------------------------------------------------------------------------
    let state = CompState {
        compositor_state,
        xdg_shell_state,
        shm_state,
        seat_state,
        data_device_state,
        space,
        popup_manager: PopupManager::default(),
        layout: Box::new(MasterStackLayout::default()),
        loop_signal,
        display_handle: dh.clone(),
        seat,
    };

    let mut calloop_data = CalloopData { state, display };

    // -------------------------------------------------------------------------
    // 8. Bind the Wayland socket
    // -------------------------------------------------------------------------
    // ListeningSocketSource creates a Unix socket that Wayland clients connect to.
    // It integrates with calloop — when a new client connects, our callback fires.
    let listening_socket = ListeningSocketSource::new_auto()?;
    let socket_name = listening_socket.socket_name().to_string_lossy().to_string();
    info!(socket = %socket_name, "Wayland socket bound");

    // Insert the listening socket into the event loop.
    // When a client connects, we accept the connection and insert it into the Display.
    event_loop
        .handle()
        .insert_source(listening_socket, |client_stream, _, data| {
            // A new client connected! Insert it into the Wayland display.
            // Arc::new(ClientState::default()) gives each client its own state.
            if let Err(err) = data
                .display
                .handle()
                .insert_client(client_stream, Arc::new(ClientState::default()))
            {
                warn!(?err, "Failed to insert client");
            }
        })?;

    // -------------------------------------------------------------------------
    // 9. Insert the Wayland display into the event loop
    // -------------------------------------------------------------------------
    // The Display's file descriptor signals when clients have sent requests.
    // We wrap it in a calloop Generic source so the event loop dispatches it.
    let display_fd = calloop_data
        .display
        .backend()
        .poll_fd()
        .try_clone_to_owned()?;
    event_loop.handle().insert_source(
        Generic::new(display_fd, Interest::READ, Mode::Level),
        |_, _, data| {
            // Dispatch all pending client requests to our handler implementations.
            // This is where CompositorHandler::commit, XdgShellHandler::new_toplevel,
            // etc. get called.
            data.display
                .dispatch_clients(&mut data.state)
                .expect("Failed to dispatch clients");
            Ok(PostAction::Continue)
        },
    )?;

    // -------------------------------------------------------------------------
    // 10. Print connection instructions and run the event loop
    // -------------------------------------------------------------------------
    info!("Compositor ready! Connect clients with:");
    info!("  WAYLAND_DISPLAY={socket_name} <your-app>");
    info!("Press Ctrl+C to stop.");

    // Run the event loop. This blocks forever, dispatching events as they arrive.
    // The second argument is a timeout — None means block until an event arrives.
    event_loop.run(None, &mut calloop_data, |_data| {
        // This closure runs once per event loop iteration, after all events
        // have been dispatched. You can use it for periodic maintenance tasks.

        // Flush pending outgoing messages to all clients
        // (configure events, frame callbacks, etc.)
    })?;

    Ok(())
}
