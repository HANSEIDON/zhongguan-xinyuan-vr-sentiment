import matplotlib.pyplot as plt
import matplotlib.patches as patches


# ==========================================================
# 1. 이 함수를 import 문 아래에 추가하세요.
# ==========================================================
def convert_xy_to_pitch_yaw(norm_x, norm_y, h_fov=180, v_fov=90):
    """
    이미지 좌표(x, y)를 Pitch와 Yaw 각도로 변환합니다.
    (x, y)는 좌측 상단이 원점(0,0)인 좌표를 기준으로 합니다.
    """

    # 2. 정규화된 좌표에 시야각(FOV)을 곱해 각도 계산
    yaw = norm_x * (h_fov / 2)
    # y좌표는 아래로 갈수록 증가하므로, pitch 계산 시 부호를 반전시켜 위를 '+'로 만듭니다.
    pitch = norm_y * (v_fov / 2)

    return pitch, yaw


# 전체 플롯 사이즈와 서브플롯 설정
# 1행 3열의 서브플롯을 생성합니다.
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
plt.rcParams["font.family"] = "D2CodingLigature Nerd Font Mono"
plt.rcParams["font.size"] = 11  # 폰트 크기 설정

# --------------------------------------------------------------------
# 1. 좌측 상단이 원점인 좌표계 (축 스타일 변경)
# --------------------------------------------------------------------
ax1.set_title("1. Raw image: Top Left is Origin (0, 0)", pad=20, weight="bold")

# 사각형 정의 (좌측 상단 (0,0)에서 너비 200, 높이 100)
rect1 = patches.Rectangle(
    (0, 0), 200, 100, linewidth=2, edgecolor="royalblue", facecolor="aliceblue"
)
ax1.add_patch(rect1)

# 원점(0,0)에 빨간 점으로 표시
ax1.plot(0, 0, "ro", markersize=10, label="Origin (0,0)")

# 좌표 텍스트 추가
ax1.text(5, 5, "(0,0)", fontsize=14, color="darkred")

# 축 범위 설정
ax1.set_xlim(-20, 220)
ax1.set_ylim(-20, 120)

# 축을 원점(0,0)에서 교차하도록 설정
ax1.spines["left"].set_position("zero")
ax1.spines["top"].set_position("zero")  # y축 기준선을 위쪽으로 변경
ax1.spines["right"].set_color("none")
ax1.spines["bottom"].set_color("none")

# 이미지 좌표계처럼 y축을 뒤집어 줍니다.
ax1.invert_yaxis()
ax1.set_aspect("equal", adjustable="box")  # 가로세로 비율을 동일하게 설정
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.set_xlabel("X-axis", loc="right")
ax1.set_ylabel("Y-axis", loc="bottom")


# --------------------------------------------------------------------
# 2. 중앙이 원점인 좌표계 (참고용)
# --------------------------------------------------------------------
ax2.set_title("2. Convert: Center as Origin (0, 0)", pad=20, weight="bold")

# 사각형 정의 (중앙이 (0,0)이 되도록 좌측 하단 좌표 설정)
rect2 = patches.Rectangle(
    (-100, -50), 200, 100, linewidth=2, edgecolor="seagreen", facecolor="honeydew"
)
ax2.add_patch(rect2)

# 원점(0,0)에 빨간 점으로 표시
ax2.plot(0, 0, "ro", markersize=10, label="Origin (0,0)")
ax2.text(5, 5, "(0,0)", fontsize=14, color="darkred")

# 축 범위 설정
ax2.set_xlim(-120, 120)
ax2.set_ylim(-70, 70)

# 축을 원점(0,0)에서 교차하도록 설정
ax2.spines["left"].set_position("zero")
ax2.spines["bottom"].set_position("zero")
ax2.spines["right"].set_color("none")
ax2.spines["top"].set_color("none")

ax2.set_aspect("equal", adjustable="box")  # 가로세로 비율을 동일하게 설정
ax2.grid(True, linestyle="--", alpha=0.6)
ax2.set_xlabel("X-axis", loc="right")
ax2.set_ylabel("Y-axis", loc="top")

# --------------------------------------------------------------------
# 3. 중앙 원점 기반, 임의의 점과 거리 표시
# --------------------------------------------------------------------
ax3.set_title("3. Object detected (x, y)", pad=20, weight="bold")

# 사각형 정의
rect3_width, rect3_height = 300, 150
rect3 = patches.Rectangle(
    (-rect3_width / 2, -rect3_height / 2),
    rect3_width,
    rect3_height,
    linewidth=2,
    edgecolor="darkorange",
    facecolor="ivory",
)
ax3.add_patch(rect3)

# 사각형 내부에 랜덤한 점 생성 (중앙이 0,0이므로 음수 범위 포함)
random_x = 98
random_y = 37
norm_x = random_x / (rect3_width / 2)
norm_y = random_y / (rect3_height / 2)

# 랜덤한 점을 파란색으로 표시
ax3.plot(random_x, random_y, "bo", markersize=10)

# 원점(0,0)에서 점까지 수직/수평선 그리기
ax3.plot([random_x, random_x], [0, random_y], color="gray", linestyle="--")
ax3.plot([0, random_x], [random_y, random_y], color="gray", linestyle="--")

# 점의 좌표 텍스트 추가
coord_text = f"Chair\n({random_x:.1f}, {random_y:.1f})"
ax3.text(
    random_x,
    random_y + 5,
    coord_text,
    fontsize=14,
    color="darkblue",
    ha="center",
    va="bottom",
)

# 축에 얼마나 떨어져 있는지 텍스트 추가
ax3.text(
    random_x,
    0,
    f"{norm_x * 100:.1f}% of X-axis",
    fontsize=12,
    color="red",
    ha="center",
    va="top",
)
ax3.text(
    0,
    random_y,
    f"{norm_y * 100:.1f}% of Y-axis",
    fontsize=12,
    color="red",
    ha="right",
    va="center",
)


# 축 범위 설정
ax3.set_xlim(-rect3_width / 2 - 20, rect3_width / 2 + 20)
ax3.set_ylim(-rect3_height / 2 - 20, rect3_height / 2 + 20)

# 축을 원점(0,0)에서 교차하도록 설정
ax3.spines["left"].set_position("zero")
ax3.spines["bottom"].set_position("zero")
ax3.spines["right"].set_color("none")
ax3.spines["top"].set_color("none")

ax3.set_aspect("equal", adjustable="box")  # 가로세로 비율을 동일하게 설정
ax3.grid(True, linestyle="--", alpha=0.6)
ax3.set_xlabel("X-axis", loc="right")
ax3.set_ylabel("Y-axis", loc="top")

# --------------------------------------------------------------------
# 4. (x,y)를 (Pitch, Yaw)로 변환한 결과 시각화
# --------------------------------------------------------------------
ax4.set_title("4. Convert into (pitch, yaw)", pad=20, weight="bold")

# 시야각(Field of View) 설정 (임의의 값)
H_FOV = 180  # 수평 시야각 120도
V_FOV = 90  # 수직 시야각 90도

# 3번 그림의 랜덤 좌표를 사용해 Pitch, Yaw 계산
# 3번 그림은 중앙이 아닌 좌측 상단이 원점이므로, 해당 사각형의 너비/높이를 사용합니다.
# (파일 코드상 3번 그림이 중앙 원점 기반이라면 rect3_width, rect3_height를 사용)
pitch, yaw = convert_xy_to_pitch_yaw(norm_x, norm_y, H_FOV, V_FOV)


# 시야각 경계를 사각형으로 표시
fov_rect = patches.Rectangle(
    (-H_FOV / 2, -V_FOV / 2),
    H_FOV,
    V_FOV,
    linewidth=2,
    edgecolor="green",
    facecolor="honeydew",
    label=f"FOV ({H_FOV}°, {V_FOV}°)",
)
ax4.add_patch(fov_rect)

# 계산된 (Yaw, Pitch) 지점에 점 찍기
ax4.plot(yaw, pitch, "ro", markersize=12)

# (0,0)에서 해당 점까지 선 긋기
ax4.plot([0, yaw], [0, pitch], "r--", alpha=0.7)

# 좌표 텍스트 추가
result_text = f"Pitch: {pitch:.2f}°\nYaw: {yaw:.2f}°"
ax4.text(
    yaw, pitch - 5, result_text, fontsize=14, color="darkred", ha="center", va="top"
)

# 축 설정
ax4.set_xlim(-H_FOV / 2 - 20, H_FOV / 2 + 20)
ax4.set_ylim(-V_FOV / 2 - 20, V_FOV / 2 + 20)
ax4.axhline(0, color="black", linewidth=0.8)
ax4.axvline(0, color="black", linewidth=0.8)
ax4.grid(True, linestyle="--", alpha=0.6)
ax4.set_aspect("equal", adjustable="box")
ax4.set_xlabel("Yaw")
ax4.set_ylabel("Pitch")
ax4.legend()


# --------------------------------------------------------------------
# 전체 레이아웃 조정 및 플롯 보이기
# --------------------------------------------------------------------

plt.tight_layout()
plt.savefig("assets\\xy_to_pitch_yaw.png", dpi=300)
