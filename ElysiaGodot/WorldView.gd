extends Node2D

var world_w:int = 256
var zoom: float = 1.0
var pan: Vector2 = Vector2.ZERO
var dragging := false
var last_mouse := Vector2.ZERO

var cells: Array = []
var show_threat := false
var show_value := true
var show_will := false
var show_coherence := false
var show_grid := true
var tex_threat: Texture2D
var tex_value: Texture2D
var tex_will: Texture2D
var tex_coherence: Texture2D
var tex_terrain: Texture2D
 var tex_veg: Texture2D
 var tex_farm: Texture2D
 var tex_river: Texture2D
 var tex_farm_paddy: Texture2D
 var tex_farm_field: Texture2D
var show_veg := true
var tex_river: Texture2D
var show_help := true
var time_phase: float = 0.0
var selected_id: String = ""
var selected_pos: Vector2 = Vector2.ZERO
var show_farm := true
 var show_help := true
 var time_phase: float = 0.0
 var selected_id: String = ''
 var selected_pos: Vector2 = Vector2.ZERO

func _ready():
	set_process(true)

func _process(_delta: float) -> void:
	var client: Node = get_node_or_null("/root/Client") as Node
	var status: Label = get_node_or_null("Status") as Label
	if client:
		if client.init_msg.has("world"):
			world_w = int(client.init_msg["world"].get("width", world_w))
		var frame: Dictionary = client.pop_frame()
		if frame.size() > 0:
			cells = frame.get("cells", cells)
			var ov: Dictionary = frame.get("overlays", {})
			tex_terrain = _decode_tex(ov.get("terrain"))
			tex_veg = _decode_tex(ov.get("veg"))
            tex_farm = _decode_tex(ov.get("farm"))
            tex_farm_paddy = _decode_tex(ov.get("farm_paddy"))
            tex_farm_field = _decode_tex(ov.get("farm_field"))
            tex_river = _decode_tex(ov.get("river"))
            tex_river = _decode_tex(ov.get("river"))
			tex_threat = _decode_tex(ov.get("threat"))
			tex_value = _decode_tex(ov.get("value"))
			tex_will = _decode_tex(ov.get("will"))
			tex_coherence = _decode_tex(ov.get("coherence"))
            var tinfo: Dictionary = frame.get("time", {})
            if tinfo.has("phase"):
                var ph = tinfo.get("phase")
                if typeof(ph) == TYPE_FLOAT or typeof(ph) == TYPE_INT:
                    time_phase = clamp(float(ph), 0.0, 1.0)
		if status:
			var msg: String = ""
			if client.connected:
				msg = "연결됨 | 셀: %d" % cells.size()
			else:
				msg = "연결 대기… (ws://127.0.0.1:8765)"
			status.text = msg
	else:
		if status:
			status.text = "Client(오토로드) 미탑재 — project.godot의 Autoload 확인"
	queue_redraw()

func _unhandled_input(event):
	if event is InputEventMouseButton:
		if event.button_index == MOUSE_BUTTON_WHEEL_UP and event.pressed:
			zoom = clamp(zoom * 1.1, 0.25, 8.0)
		elif event.button_index == MOUSE_BUTTON_WHEEL_DOWN and event.pressed:
			zoom = clamp(zoom / 1.1, 0.25, 8.0)
		elif event.button_index == MOUSE_BUTTON_MIDDLE:
			dragging = event.pressed
			last_mouse = event.position
		elif event.button_index == MOUSE_BUTTON_LEFT and event.pressed:
			var client: Node = get_node_or_null("/root/Client") as Node
			if client:
				var coords: Vector2 = _screen_to_world(event.position)
				client.send_input({"select_x": int(coords.x), "select_y": int(coords.y)})
	elif event is InputEventMouseMotion and dragging:
		var d: Vector2 = event.position - last_mouse
		pan -= d
		last_mouse = event.position
	elif event is InputEventKey and event.pressed:
		match event.keycode:
			KEY_T: show_threat = !show_threat
			KEY_V: show_value = !show_value
			KEY_I: show_will = !show_will
			KEY_U: show_coherence = !show_coherence
			KEY_G: show_grid = !show_grid
			KEY_Y: show_veg = !show_veg
			KEY_F: show_farm = !show_farm
\t\t\tKEY_H: show_help = !show_help
			KEY_F9:
				var client: Node = get_node_or_null("/root/Client") as Node
				if client:
					var p: Vector2 = get_viewport().get_mouse_position()
					var coords: Vector2 = _screen_to_world(p)
					client.send_input({"disaster": {"kind":"FLOOD","x":int(coords.x),"y":int(coords.y),"radius":8}})
			KEY_F10:
				var client2: Node = get_node_or_null("/root/Client") as Node
				if client2:
					var p2: Vector2 = get_viewport().get_mouse_position()
					var coords2: Vector2 = _screen_to_world(p2)
					client2.send_input({"disaster": {"kind":"VOLCANO","x":int(coords2.x),"y":int(coords2.y),"radius":8}})

func _screen_to_world(p: Vector2) -> Vector2:
	var vp: Vector2 = get_viewport_rect().size
	var base: float = float(min(vp.x, vp.y))
	var s: float = (base / float(world_w)) * zoom
	var off: Vector2 = Vector2((vp.x - world_w*s)/2.0, (vp.y - world_w*s)/2.0)
	var tl: Vector2 = off + pan
	var wx: float = (p.x - tl.x) / s
	var wy: float = (p.y - tl.y) / s
	return Vector2(clamp(wx, 0.0, world_w-1.0), clamp(wy, 0.0, world_w-1.0))

func _decode_tex(b64):
	if typeof(b64) != TYPE_STRING or b64 == "":
		return null
	var buf: PackedByteArray = Marshalls.base64_to_raw(b64)
	var img := Image.new()
	var err := img.load_png_from_buffer(buf)
	if err != OK:
		return null
	return ImageTexture.create_from_image(img)

func _draw():
	var vp: Vector2 = get_viewport_rect().size
	var base: float = float(min(vp.x, vp.y))
	var s: float = (base / float(world_w)) * zoom
	var off: Vector2 = Vector2((vp.x - world_w*s)/2.0, (vp.y - world_w*s)/2.0)
	var tl: Vector2 = off + pan
	# Softer dark background then terrain base
	var grad := GradientTexture2D.new()\n    var g := Gradient.new()\n    g.colors = PackedColorArray([_sky_color_top(time_phase), _sky_color_bottom(time_phase)])\n    grad.gradient = g\n    grad.width = int(vp.x)\n    grad.height = int(vp.y)\n    draw_texture_rect(grad, Rect2(Vector2.ZERO, vp), false)
	var rect: Rect2 = Rect2(tl, Vector2(world_w*s, world_w*s))
	if tex_terrain != null:
		draw_texture_rect(tex_terrain, rect, false)
    if tex_river != null:
        draw_texture_rect(tex_river, rect, false, Color(0.4,0.7,1.0,0.45))
	# Optional grid overlay
	if show_grid:
		var step_world: int = max(8, world_w / 32)
		var x: int = 0
		while x <= world_w:
			var xp: float = tl.x + float(x) * s
			draw_line(Vector2(xp, tl.y), Vector2(xp, tl.y + world_w*s), Color(1,1,1,0.08), 1.0)
			x += step_world
		var y: int = 0
		while y <= world_w:
			var yp: float = tl.y + float(y) * s
			draw_line(Vector2(tl.x, yp), Vector2(tl.x + world_w*s, yp), Color(1,1,1,0.08), 1.0)
			y += step_world
	if show_veg and tex_veg != null:
		draw_texture_rect(tex_veg, rect, false, Color(0.5,1.0,0.6,0.30))
	if show_farm and tex_farm != null:
		draw_texture_rect(tex_farm, rect, false, Color(1.0,0.9,0.4,0.35))
	if show_threat and tex_threat != null:
		draw_texture_rect(tex_threat, rect, false, Color(1,0.2,0.1,0.35))
	if show_value and tex_value != null:
		draw_texture_rect(tex_value, rect, false, Color(1.0,0.9,0.4,0.35))
	if show_will and tex_will != null:
		draw_texture_rect(tex_will, rect, false, Color(0.3,0.6,1.0,0.35))
	if show_coherence and tex_coherence != null:
		draw_texture_rect(tex_coherence, rect, false, Color(1.0,0.95,0.6,0.35))
        var hover: Dictionary = _nearest_cell_at_screen(get_viewport().get_mouse_position())
	for c in cells:
		var cd: Dictionary = c
		var x: float = float(cd.get("x",0.0))
		var y: float = float(cd.get("y",0.0))
		var alive: bool = bool(cd.get("alive", true))
		var typ: String = str(cd.get("type", ""))
		var col: Color
		if typ == "human":
			col = Color(0.90, 0.85, 0.50)
		elif typ == "animal":
			col = Color(0.50, 0.70, 1.00)
		elif typ == "life":
			col = Color(0.50, 0.90, 0.55)
		else:
			col = Color(0.85, 0.85, 0.90)
		if not alive:
			col = Color(col.r*0.5, col.g*0.5, col.b*0.5, 1.0)
		var p: Vector2 = tl + Vector2(x*s, y*s)
		var radius: float = clamp(3.0*s, 2.0, 6.0)
        if hover.size() > 0 and String(hover.get("id","")) == String(cd.get("id","")):
            draw_circle(p, radius+3.0, Color(1,1,1,0.6), 2.0)
        if selected_id != "" and String(cd.get("id","")) == selected_id:
            draw_circle(p, radius+5.0, Color(0.4,0.8,1.0,0.9), 2.5)
		draw_circle(p, radius, col)
    if show_help:
        var lines := [
            "키: T 위협, V 가치, I 의지, Y 식생, F 농경, G 그리드, H 도움말",
            "재해: F9 범람, F10 화산 — 마우스 위치",
            "카메라: 휠 줌, 중클릭 드래그 이동, 좌클릭 선택",
        ]
        var y0 := 12.0
        for l in lines:
            draw_string(get_theme_default_font(), Vector2(12, y0), l, HAlign.LEFT, -1.0, 16.0, Color(0.92,0.94,0.96,0.95))
            y0 += 18.0
		draw_circle(p, radius, col)
