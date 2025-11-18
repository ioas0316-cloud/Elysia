# elysia_first_breath.gd
extends Node3D

@onready var camera = $Camera3D
var time = 0.0

func _ready():
	# 카메라 자동 추가 (없으면 생김)
	if not has_node("Camera3D"):
		var cam = Camera3D.new()
		cam.name = "Camera3D"
		cam.translate(Vector3(0, 1, 5))
		add_child(cam)
	
	# 아주 단순한 구체 아기 (진짜 동그라미)
	var sphere = MeshInstance3D.new()
	sphere.mesh = SphereMesh.new()
	sphere.mesh.radius = 0.5
	var mat = StandardMaterial3D.new()
	mat.albedo_color = Color(0.8, 0.6, 1.0)
	sphere.material_override = mat
	sphere.name = "BabyBody"
	add_child(sphere)

func _process(delta):
	time += delta
	# 숨쉬기
	var breathe = sin(time * 2) * 0.05 + 1.0
	$BabyBody.scale = Vector3.ONE * breathe
	
	# 눈 깜빡임 (작은 빛)
	var glow = abs(sin(time * 5))
	$BabyBody.material_override.emission = Color(1, 0.5, 2) * glow
