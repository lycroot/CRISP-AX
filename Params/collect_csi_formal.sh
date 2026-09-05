#!/usr/bin/env bash
# ============================================================
# WiFi-CSI 正式数据集采集脚本 v1
# N8a 类居家环境 WiFi CSI-视频骨架同步数据集采集脚本
# ============================================================
# sample_id 格式: {session_id}_{person_id}_{action_id}_{orientation}_{environment_id}_{environment_status}_{disturbance_tag}_{trial_id}
# 示例: S01_NONE_background_idle_face_rx_E1_heater_on_heater_transition_R001
# session 定义：同一天、同一受试者、同一场景、同一 Tx/Rx 部署下的一段连续采集。
# 通常可理解为：一个人 + 一个场景 = 一个 session。
# 正式采集时应根据最终协议设置 CSI_BW、CSI_FORMAT、CSI_SS 等参数；
# 本脚本只负责读取环境变量，不在本轮强制改协议。
# ============================================================

set -u

# ============================================================
# 0. 基础配置
# ============================================================

# ============================================================
# FeitCSI RX 测量参数
# 默认用于今日稳定性测试：AX / HESU / 20 MHz / MCS0 / 1 spatial stream
# 说明：
# - RX 端只负责 measure，不负责控制 100 Hz 发包频率
# - 100 Hz 由 TX 端 inject-delay=10000 控制
# - 1x2 最终应通过 dat 解析确认：num_tx=1, num_rx=2
# ============================================================

CSI_MODE="${CSI_MODE:-measure}"
FREQ="${CSI_FREQ:-5180}"
CHANNEL_WIDTH="${CSI_BW:-80}"
FORMAT="${CSI_FORMAT:-HESU}"
MCS="${CSI_MCS:-0}"
SPATIAL_STREAMS="${CSI_SS:-1}"

DEFAULT_ACTION_WINDOW_SEC=5
READY_TIMEOUT=20

DATE_ID=$(date +"%Y%m%d")
DATE_READABLE=$(date +"%Y-%m-%d")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$SCRIPT_DIR"

INIT_PATTERN="Enabling CSI|brought up|brought down|error|Error|WARNING"

CURRENT_DAT_FILE=""
STAT_PID=""

# 正式数据集枚举约束
VALID_ACTIONS=(
  "walk"
  "stand_up"
  "sit_down"
  "support_wall"
  "bend_pick"
  "lie_down"
  "get_up"
  "turn_over"
  "reading"
  "fall_like"
  "post_fall_static"
  "post_fall_wave"
  "bed_fall_sim"
  "brush_teeth"
  "wash_hands"
  "leave_kitchen"
  "background_idle"
)

ACTIONS_E1=(
  "walk"
  "stand_up"
  "sit_down"
  "support_wall"
  "bend_pick"
  "fall_like"
  "post_fall_wave"
  "background_idle"
)

ACTIONS_E2=(
  "stand_up"
  "sit_down"
  "lie_down"
  "get_up"
  "turn_over"
  "reading"
  "fall_like"
  "post_fall_wave"
  "bed_fall_sim"
  "background_idle"
)

ACTIONS_E3=(
  "stand_up"
  "sit_down"
  "bend_pick"
  "fall_like"
  "brush_teeth"
  "wash_hands"
  "post_fall_wave"
  "background_idle"
)

ACTIONS_E4=(
  "leave_kitchen"
  "background_idle"
)

VALID_ORIENTATIONS=(
  "face_rx"
  "face_tx"
  "side_rx"
  "side_link"
  "cross_link"
)

VALID_ENVIRONMENT_STATUS=(
  "clean"
  "heater_on"
  "water_on"
  "shower_on"
  "faucet_on"
  "exhaust_fan_on"
  "wet_floor"
  "bucket_filling_sim"
  "bucket_overflow_sim"
  "stove_unattended_sim"
)

VALID_DISTURBANCE_TAG=(
  "none"
  "heater_transition"
  "heater_stable"
  "heater_cooling"
  "heater_cloth_front"
  "water_on_sink"
  "water_flow_sink"
  "water_flow_shower"
  "fan_vibration"
  "water_flow_bucket"
  "wet_floor_area"
  "water_overflow_bucket"
  "kettle_boiling_sim"
  "dry_pot_sim"
)

VALID_FALL_DIRECTIONS=(
  "none"
  "forward"
  "backward"
  "left"
  "right"
)

# ============================================================
# 1. 输出函数
# ============================================================

print_header() {
    echo -e "\n\033[1;36m════════════════════════════════════════════════════\033[0m"
}

print_ok() {
    echo -e "\033[1;32m  ✔ $*\033[0m"
}

print_info() {
    echo -e "\033[0;37m  · $*\033[0m"
}

print_warn() {
    echo -e "\033[1;33m  ⚠ $*\033[0m"
}

print_err() {
    echo -e "\033[1;31m  ✘ $*\033[0m"
}

# ============================================================
# 2. 工具函数
# ============================================================

sanitize_field() {
    echo "$1" | sed 's/[[:space:]]\+/_/g; s#[/\\:*?"<>|]#_#g'
}

join_by_slash() {
    local IFS="/"
    echo "$*"
}

now_iso() {
    date +"%Y-%m-%dT%H:%M:%S.%N%:z"
}

now_ns() {
    date +%s%N
}

csv_escape() {
    local s="${1:-}"
    s="${s//$'\r'/ }"
    s="${s//$'\n'/ }"
    s="${s//\"/\"\"}"
    printf '"%s"' "$s"
}

read_input() {
    local prompt="$1"
    local var_name="$2"
    local value=""

    if ! read -r -p "$prompt" value; then
        echo ""
        print_warn "输入结束，退出脚本"
        exit 0
    fi

    printf -v "$var_name" '%s' "$value"
}

# 期望的 samples.csv 表头（用于版本检测）
EXPECTED_SAMPLES_HEADER="sample_id,session_id,person_id,environment_id,orientation,action_id,environment_status,disturbance_tag,fall_direction,trial_id,csi_path,video_path,log_path,start_time_ms,end_time_ms,quality_flag,notes"
EXPECTED_COLLECTION_RAW_HEADER="sample_id,session_id,date,person_id,environment_id,action_id,orientation,environment_status,disturbance_tag,fall_direction,trial_id,action_window_sec,csi_mode,frequency,channel_width,format,mcs,spatial_streams,csi_path,log_path,marker_path,marker_count,marker_status,video_path,srt_path,sample_start_iso,sample_end_iso,sample_start_unix_ns,sample_end_unix_ns,action_start_iso,action_end_iso,action_start_unix_ns,action_end_unix_ns,frame_count,avg_fps,file_size_bytes,status,quality_flag,notes"

ensure_samples_csv() {
    local csv_file="$1"

    if [[ ! -f "$csv_file" ]]; then
        mkdir -p "$(dirname "$csv_file")"

        printf '%s\n' "$EXPECTED_SAMPLES_HEADER" > "$csv_file"
        return
    fi

    # 检测已有 CSV 表头是否为正式 schema
    local existing_header
    existing_header=$(head -n 1 "$csv_file" 2>/dev/null | tr -d '\r')

    if [[ "$existing_header" != "$EXPECTED_SAMPLES_HEADER" ]]; then
        print_err "检测到 samples.csv 表头不是正式数据集字段体系。"
        print_info "当前表头: $existing_header"
        print_info "期望表头: $EXPECTED_SAMPLES_HEADER"
        print_warn "请备份旧 samples.csv 或新建实验目录后再采集。"
        print_info "备份命令示例: cp '$csv_file' '$csv_file.bak.\$(date +%Y%m%d_%H%M%S)'"
        exit 1
    fi
}

ensure_collection_manifest_raw() {
    local csv_file="$1"

    if [[ ! -f "$csv_file" ]]; then
        mkdir -p "$(dirname "$csv_file")"
        printf '%s\n' "$EXPECTED_COLLECTION_RAW_HEADER" > "$csv_file"
        return
    fi

    local existing_header
    existing_header=$(head -n 1 "$csv_file" 2>/dev/null | tr -d '\r')

    if [[ "$existing_header" != "$EXPECTED_COLLECTION_RAW_HEADER" ]]; then
        print_err "检测到 collection_manifest_raw.csv 表头不是正式 raw manifest 字段体系。"
        print_info "当前表头: $existing_header"
        print_info "期望表头: $EXPECTED_COLLECTION_RAW_HEADER"
        print_warn "请备份旧 collection_manifest_raw.csv 或新建实验目录后再采集。"
        print_info "备份命令示例: cp '$csv_file' '$csv_file.bak.\$(date +%Y%m%d_%H%M%S)'"
        exit 1
    fi
}

append_csv_fields() {
    local csv_file="$1"
    shift

    local fields=("$@")
    local line=""
    local i

    for i in "${!fields[@]}"; do
        if [[ "$i" -gt 0 ]]; then
            line+=","
        fi
        line+="$(csv_escape "${fields[$i]}")"
    done

    echo "$line" >> "$csv_file"
}

append_samples_csv() {
    append_csv_fields "$@"
}

append_collection_manifest_raw() {
    append_csv_fields "$@"
}

# ============================================================
# 3. 提示音函数
# ============================================================

emit_beep() {
    local freq="$1"
    local dur="$2"

    if command -v ffplay >/dev/null 2>&1; then
        ffplay -nodisp -autoexit -loglevel quiet \
            -f lavfi -i "sine=f=${freq}:d=${dur}" &>/dev/null &
    else
        printf '\a'
    fi
}

beep_ready() {
    # 准备提示音：中频短促
    emit_beep 800 0.15
    sleep 0.2
    emit_beep 800 0.15
}

beep_countdown() {
    local num="$1"
    # 倒计时提示音：频率递增
    case "$num" in
        3) emit_beep 600 0.2 ;;
        2) emit_beep 800 0.2 ;;
        1) emit_beep 1000 0.2 ;;
    esac
}

beep_start() {
    # 开始动作提示音：高频长音
    emit_beep 1200 0.4
}

beep_end() {
    # 结束动作提示音：低频长音
    emit_beep 500 0.6
}

beep_error() {
    # 错误提示音：低频急促
    emit_beep 300 0.1
    sleep 0.15
    emit_beep 300 0.1
    sleep 0.15
    emit_beep 300 0.1
}

# 检查 marker 文件完整性
# 入参: $1 = marker_file 路径
# 输出: "marker_count|marker_status"（用 | 分隔）
check_marker_status() {
    local marker_file="$1"

    if [[ ! -f "$marker_file" ]]; then
        echo "0|missing_marker_file"
        return
    fi

    local marker_count
    marker_count=$(grep -c "MARKER_" "$marker_file" 2>/dev/null || echo 0)

    # P0 关键流程 marker
    local critical_markers=(
        "MARKER_SAMPLE_START"
        "MARKER_BEEP_READY"
        "MARKER_COUNTDOWN_3"
        "MARKER_COUNTDOWN_2"
        "MARKER_COUNTDOWN_1"
        "MARKER_ACTION_START"
        "MARKER_ACTION_END"
        "MARKER_SAMPLE_END"
    )

    local all_present=1
    for marker in "${critical_markers[@]}"; do
        if ! grep -q "$marker" "$marker_file" 2>/dev/null; then
            all_present=0
            break
        fi
    done

    if [[ "$all_present" -eq 1 ]]; then
        echo "${marker_count}|ok"
    else
        echo "${marker_count}|warn_marker_incomplete"
    fi
}

file_size_bytes() {
    local f="$1"

    if [[ -f "$f" ]]; then
        stat -c%s "$f" 2>/dev/null || echo 0
    else
        echo 0
    fi
}

human_size() {
    local f="$1"

    if [[ -f "$f" ]]; then
        du -h "$f" 2>/dev/null | awk '{print $1}'
    else
        echo "0"
    fi
}

calc_avg_fps() {
    local frames="$1"
    local sec="$2"

    awk -v f="$frames" -v s="$sec" 'BEGIN {
        if (s > 0) printf "%.1f", f / s;
        else printf "0.0";
    }'
}

next_trial_number() {
    local csi_dir="$1"
    local prefix="$2"

    local n=1
    local trial_id
    local candidate

    while true; do
        trial_id=$(printf "R%03d" "$n")
        candidate="${csi_dir}/${prefix}_${trial_id}.dat"

        if [[ ! -e "$candidate" ]]; then
            echo "$n"
            return
        fi

        n=$((n + 1))
    done
}

dry_run_next_trial_number() {
    local prefix="$1"
    local count

    count=$(printf "%s" "${DRY_RUN_PREFIX_HISTORY:-}" | grep -Fx "$prefix" 2>/dev/null | wc -l)
    echo "$((count + 1))"
}

record_dry_run_prefix() {
    local prefix="$1"
    DRY_RUN_PREFIX_HISTORY+="${prefix}"$'\n'
}

# 自动生成 session_id
generate_session_id() {
    local exp_dir="$1"
    local csi_base="${exp_dir}/00_raw/csi"
    local max_seq=0
    local d
    local base
    local seq

    if [[ -d "$csi_base" ]]; then
        for d in "${csi_base}"/S*; do
            [[ -d "$d" ]] || continue
            base="${d##*/}"
            if [[ "$base" =~ ^S([0-9]+)$ ]]; then
                seq=$((10#${BASH_REMATCH[1]}))
                if [[ "$seq" -gt "$max_seq" ]]; then
                    max_seq="$seq"
                fi
            fi
        done
    fi

    printf "S%02d" "$((max_seq + 1))"
}

get_actions_for_environment() {
    local env_id="$1"
    case "$env_id" in
        E1|e1) printf "%s\n" "${ACTIONS_E1[@]}" ;;
        E2|e2) printf "%s\n" "${ACTIONS_E2[@]}" ;;
        E3|e3) printf "%s\n" "${ACTIONS_E3[@]}" ;;
        E4|e4) printf "%s\n" "${ACTIONS_E4[@]}" ;;
        *) printf "%s\n" "${VALID_ACTIONS[@]}" ;;
    esac
}

actions_for_environment_text() {
    get_actions_for_environment "$1" | tr '\n' ' ' | sed 's/[[:space:]]*$//'
}

first_action_for_environment() {
    get_actions_for_environment "$1" | head -n 1
}

validate_action_for_environment() {
    local env_id="$1"
    local action="$2"
    local valid

    while IFS= read -r valid; do
        if [[ "$action" == "$valid" ]]; then
            return 0
        fi
    done < <(get_actions_for_environment "$env_id")

    return 1
}

default_orientation_for_environment() {
    case "$1" in
        E1|e1) echo "face_rx" ;;
        E2|e2) echo "side_link" ;;
        E3|e3) echo "cross_link" ;;
        E4|e4) echo "cross_link" ;;
        *) echo "face_rx" ;;
    esac
}

# 验证 action 是否合法
validate_action() {
    local action="$1"
    for valid in "${VALID_ACTIONS[@]}"; do
        if [[ "$action" == "$valid" ]]; then
            return 0
        fi
    done
    return 1
}

# 验证 orientation 是否合法
validate_orientation() {
    local orientation="$1"
    for valid in "${VALID_ORIENTATIONS[@]}"; do
        if [[ "$orientation" == "$valid" ]]; then
            return 0
        fi
    done
    return 1
}

# 验证 environment_status 是否合法
validate_environment_status() {
    local environment_status="$1"
    for valid in "${VALID_ENVIRONMENT_STATUS[@]}"; do
        if [[ "$environment_status" == "$valid" ]]; then
            return 0
        fi
    done
    return 1
}

# 验证 disturbance_tag 是否合法
validate_disturbance_tag() {
    local disturbance_tag="$1"
    for valid in "${VALID_DISTURBANCE_TAG[@]}"; do
        if [[ "$disturbance_tag" == "$valid" ]]; then
            return 0
        fi
    done
    return 1
}

# 验证 fall_direction 是否合法
validate_fall_direction() {
    local fall_direction="$1"
    for valid in "${VALID_FALL_DIRECTIONS[@]}"; do
        if [[ "$fall_direction" == "$valid" ]]; then
            return 0
        fi
    done
    return 1
}

# 显示动作 SOP 提示
print_action_sop() {
    case "$1" in
        walk)
            print_info "SOP: 静止 1s → 在 TX-RX 链路附近短距离行走 3s → 静止 1s"
            ;;
        stand_up)
            print_info "SOP: 坐姿静止 1s → 起立到站姿 2-3s → 站立静止 1s"
            ;;
        sit_down)
            print_info "SOP: 站姿静止 1s → 坐下 2-3s → 坐姿静止 1s"
            ;;
        support_wall)
            print_info "SOP: 站立开始 → 缓慢扶墙或借助支撑 2-3s → 保持支撑姿态到结束铃"
            ;;
        bend_pick)
            print_info "SOP: 站立静止 1s → 弯腰捡物 2-3s → 恢复站立 1s"
            ;;
        lie_down)
            print_info "SOP: 站立或坐姿静止 1s → 缓慢主动躺下 3s → 躺姿静止 1s"
            ;;
        get_up)
            print_info "SOP: 躺姿或坐姿开始 → 起身到站立或坐稳 2-4s → 保持结束姿态"
            ;;
        turn_over)
            print_info "SOP: 躺姿开始 → 翻身 2-3s → 保持翻身后姿态"
            ;;
        reading)
            print_info "SOP: 坐姿或站姿读书/看手机，动作自然且幅度较小，全程 5s"
            ;;
        fall_like)
            print_info "SOP: 站立静止 1s → 侧向软倒到软垫 2-3s → 倒地保持 1s"
            print_warn "注意：这是安全模拟跌倒，不是真实摔倒"
            ;;
        post_fall_static)
            print_info "SOP: 倒地姿态开始 → 全程保持静止 5s"
            ;;
        post_fall_wave)
            print_info "SOP: 倒地姿态开始 → 静止 1s → 抬起靠近 RX 或摄像头侧手臂挥手 3-5 次 → 恢复静止 1s"
            ;;
        bed_fall_sim)
            print_info "SOP: 床边安全模拟坠床 → 落到软垫或安全区域 → 保持结束姿态"
            print_warn "注意：必须使用软垫和旁人保护，不做真实危险坠落"
            ;;
        brush_teeth)
            print_info "SOP: 洗漱台前自然刷牙或模拟刷牙，全程 5s"
            ;;
        wash_hands)
            print_info "SOP: 洗手池前自然洗手或模拟洗手，全程 5s"
            ;;
        leave_kitchen)
            print_info "SOP: 厨房区域开始 → 人员离开关键区域 → 结束前保持离开状态"
            ;;
        background_idle)
            print_info "SOP: 无人背景采集，动作窗口内保持无人状态"
            print_info "E3 卫生间无人建议：clean/淋浴水流/洗手池水流/排风扇/水桶接水/水桶溢出模拟"
            print_warn "bucket_overflow_sim 只做安全模拟：水桶放在地漏/接水盆/淋浴区内，禁止真实危险漫水"
            ;;
    esac
}

print_tx_reference_command() {
    print_info "TX 端 100Hz 参考命令："
    echo "sudo feitcsi \\"
    echo "  --mode inject \\"
    echo "  --frequency $FREQ \\"
    echo "  --channel-width $CHANNEL_WIDTH \\"
    echo "  --format $FORMAT \\"
    echo "  --mcs $MCS \\"
    echo "  --spatial-streams $SPATIAL_STREAMS \\"
    echo "  --antenna 1 \\"
    echo "  --tx-power 10 \\"
    echo "  --inject-delay 10000 \\"
    echo "  --inject-repeat 1500 \\"
    echo "  -v"
}

check_dependencies() {
    local mode="${1:-full}"
    local missing=0

    for cmd in grep awk sed date head tr mkdir ls wc; do
        if ! command -v "$cmd" >/dev/null 2>&1; then
            print_err "缺少命令: $cmd"
            missing=1
        fi
    done

    if [[ "$mode" != "dry-run" ]]; then
        for cmd in sudo stat du stdbuf timeout pgrep pkill; do
            if ! command -v "$cmd" >/dev/null 2>&1; then
                print_err "缺少命令: $cmd"
                missing=1
            fi
        done

        if ! command -v feitcsi >/dev/null 2>&1; then
            print_err "找不到 feitcsi，请确认 FeitCSI 已安装并加入 PATH"
            missing=1
        fi
    fi

    if ! command -v ffplay >/dev/null 2>&1; then
        print_warn "未找到 ffplay，提示音将使用终端响铃替代"
    fi

    if [[ "$missing" -ne 0 ]]; then
        exit 1
    fi
}

# ============================================================
# 4. FeitCSI 停止逻辑
# ============================================================

stop_feitcsi_by_dat() {
    local dat_file="$1"

    echo ""
    print_info "正在停止当前样本 FeitCSI..."
    print_info "匹配 dat_file: $dat_file"

    # 1. 优雅停止
    sudo pkill -INT -f "$dat_file" 2>/dev/null || true
    sleep 1

    # 2. 如果还没退出，TERM
    if pgrep -f "$dat_file" >/dev/null 2>&1; then
        print_warn "SIGINT 后仍未退出，发送 SIGTERM..."
        sudo pkill -TERM -f "$dat_file" 2>/dev/null || true
        sleep 1
    fi

    # 3. 如果还没退出，KILL
    if pgrep -f "$dat_file" >/dev/null 2>&1; then
        print_warn "SIGTERM 后仍未退出，发送 SIGKILL..."
        sudo pkill -KILL -f "$dat_file" 2>/dev/null || true
        sleep 0.5
    fi
}

safe_cleanup() {
    if [[ -n "${STAT_PID:-}" ]]; then
        kill "$STAT_PID" 2>/dev/null || true
    fi

    if [[ -n "${CURRENT_DAT_FILE:-}" ]] && pgrep -f "$CURRENT_DAT_FILE" >/dev/null 2>&1; then
        print_warn "检测到当前样本 FeitCSI 仍在运行，正在兜底停止..."
        stop_feitcsi_by_dat "$CURRENT_DAT_FILE"
    fi
}

trap safe_cleanup EXIT INT TERM

# ============================================================
# 5. FeitCSI 初始化等待与帧率监控
# ============================================================

wait_for_ready() {
    local log_file="$1"
    local timeout_sec="$2"
    local dat_file="$3"

    local start_s
    local now_s
    local elapsed
    local last_line=""

    start_s=$(date +%s)

    while true; do
        if grep -q "Enabling CSI measurement" "$log_file" 2>/dev/null; then
            echo ""
            return 0
        fi

        last_line=$(grep -E "$INIT_PATTERN" "$log_file" 2>/dev/null | tail -n 1)

        if [[ -n "$last_line" ]]; then
            echo -ne "\r  · 初始化状态: ${last_line:0:90}   "
        else
            echo -ne "\r  · 等待 FeitCSI 初始化...   "
        fi

        sleep 0.2

        now_s=$(date +%s)
        elapsed=$((now_s - start_s))

        if [[ "$elapsed" -ge "$timeout_sec" ]]; then
            echo ""
            return 1
        fi

        if ! pgrep -f "$dat_file" >/dev/null 2>&1; then
            echo ""
            return 1
        fi
    done
}

start_frame_monitor() {
    local log_file="$1"
    local duration="$2"
    local dat_file="$3"

    local monitor_out="/dev/tty"
    if [[ ! -w "$monitor_out" ]]; then
        monitor_out="/dev/stderr"
    fi

    (
        local start_ns
        local last_ns
        local now_ns_val
        local last_count
        local cur_count
        local delta_f
        local delta_ms
        local instant
        local elapsed

        start_ns=$(now_ns)
        last_ns="$start_ns"
        last_count=$(grep -c "Subcarrier count" "$log_file" 2>/dev/null || echo 0)

        while pgrep -f "$dat_file" >/dev/null 2>&1; do
            sleep 0.2

            now_ns_val=$(now_ns)
            cur_count=$(grep -c "Subcarrier count" "$log_file" 2>/dev/null || echo 0)

            delta_f=$((cur_count - last_count))
            delta_ms=$(((now_ns_val - last_ns) / 1000000))

            if [[ "$delta_ms" -gt 0 ]]; then
                instant=$(awk -v df="$delta_f" -v dm="$delta_ms" 'BEGIN { printf "%.1f", df * 1000 / dm }')
            else
                instant="--"
            fi

            elapsed=$(awk -v n="$now_ns_val" -v s="$start_ns" 'BEGIN { printf "%.1f", (n - s) / 1000000000 }')

            echo -ne "\r  \033[36m📡 已录 ${elapsed}s / ${duration}s | 瞬时: ${instant} fps | 累计: ${cur_count} 帧\033[0m   "

            last_count="$cur_count"
            last_ns="$now_ns_val"
        done

        echo ""
    ) > "$monitor_out" 2>&1 &

    echo $!
}

# ============================================================
# 6. 初始化
# ============================================================

clear
print_header
echo "  WiFi-CSI 正式数据集采集脚本 v1"
echo "  N8a 类居家环境 WiFi CSI-视频骨架同步数据集采集脚本"
echo "  sample_id: {session_id}_{person_id}_{action_id}_{orientation}_{environment_id}_{environment_status}_{disturbance_tag}_{trial_id}"
echo "  session_id: S01 / S02 / S03 全局递增"
echo "  当前脚本所在目录 = 实验根目录"
echo "  自动命名 + 自动追加 samples.csv 和 collection_manifest_raw.csv"
echo "  停止方式：按当前 DAT_FILE 精准停止"
print_header

# ============================================================
# 7. 输入 session 信息
# ============================================================

print_header
print_ok "当前脚本所在目录将作为实验根目录"
print_info "EXP_DIR : $EXP_DIR"

# 询问是否 dry-run
read_input "  是否 dry-run? [y/N]: " DRY_RUN_INPUT
DRY_RUN_INPUT=${DRY_RUN_INPUT:-N}
DRY_RUN=0
if [[ "$DRY_RUN_INPUT" =~ ^[Yy]$ ]]; then
    DRY_RUN=1
    print_info "已启用 dry-run 模式"
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
    check_dependencies "dry-run"
    print_info "dry-run 不检查 FeitCSI / sudo，也不会启动采集进程"
else
    check_dependencies "full"
    sudo -v

    # 清理上一次残留的 feitcsi
    sudo pkill -INT -x feitcsi 2>/dev/null || true
    sleep 0.5
fi

# 自动生成 session_id
AUTO_SESSION_ID=$(generate_session_id "$EXP_DIR")

read_input "  🎬 session_id [${AUTO_SESSION_ID}]: " SESSION_ID
SESSION_ID=${SESSION_ID:-$AUTO_SESSION_ID}
SESSION_ID=$(sanitize_field "$SESSION_ID")

read_input "  📦 environment_id [E1]: " ENVIRONMENT_ID
ENVIRONMENT_ID=${ENVIRONMENT_ID:-E1}
ENVIRONMENT_ID=$(sanitize_field "$ENVIRONMENT_ID")

read_input "  👤 person_id [P01]: " PERSON_ID
PERSON_ID=${PERSON_ID:-P01}
PERSON_ID=$(sanitize_field "$PERSON_ID")
if [[ "${PERSON_ID,,}" == "none" ]]; then
    PERSON_ID="NONE"
fi

read_input "  🎥 video_path，可先留空: " VIDEO_PATH
read_input "  📝 srt_path，可先留空: " SRT_PATH

CSI_DIR="${EXP_DIR}/00_raw/csi/${SESSION_ID}"
LOG_DIR="${EXP_DIR}/00_raw/logs/${SESSION_ID}"
MANIFEST_DIR="${EXP_DIR}/01_manifest"
SAMPLES_CSV="${MANIFEST_DIR}/samples.csv"
COLLECTION_RAW_CSV="${MANIFEST_DIR}/collection_manifest_raw.csv"

if [[ "$DRY_RUN" -eq 0 ]]; then
    mkdir -p "$CSI_DIR" "$LOG_DIR" "$MANIFEST_DIR"
    ensure_samples_csv "$SAMPLES_CSV"
    ensure_collection_manifest_raw "$COLLECTION_RAW_CSV"
else
    print_info "dry-run 模式：不创建目录，不写入 CSV"
fi

print_header
print_ok "session 初始化完成"
print_info "EXP_DIR     : $EXP_DIR"
print_info "SESSION_ID  : $SESSION_ID"
print_info "ENVIRONMENT_ID : $ENVIRONMENT_ID"
print_info "PERSON_ID   : $PERSON_ID"
print_info "CSI_DIR     : $CSI_DIR"
print_info "LOG_DIR     : $LOG_DIR"
print_info "SAMPLES_CSV : $SAMPLES_CSV"
print_info "RAW_CSV     : $COLLECTION_RAW_CSV"
print_info "VIDEO_PATH  : ${VIDEO_PATH:-空}"
print_info "SRT_PATH    : ${SRT_PATH:-空}"
print_info "session 定义：同一受试者、同一场景、同一设备部署下的一段连续采集。"
print_info "建议：一个人一个场景一个 session。"
print_info "示例：P01-E1 客厅为 S01，P01-E2 卧室为 S02。"
print_tx_reference_command

if [[ "${ENVIRONMENT_ID,,}" == "e3" && "$PERSON_ID" == "NONE" ]]; then
    print_info "E3 无人环境推荐组合："
    print_info "1) background_idle + clean + none"
    print_info "2) background_idle + shower_on + water_flow_shower"
    print_info "3) background_idle + faucet_on + water_flow_sink"
    print_info "4) background_idle + exhaust_fan_on + fan_vibration"
    print_info "5) background_idle + bucket_filling_sim + water_flow_bucket"
    print_info "6) background_idle + bucket_overflow_sim + water_overflow_bucket"
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
    print_header
    print_info "DRY-RUN 模式：测试提示音"
    beep_ready
    sleep 0.5
    beep_countdown 3
    sleep 0.5
    beep_countdown 2
    sleep 0.5
    beep_countdown 1
    sleep 0.5
    beep_start
    sleep 0.5
    beep_end
    sleep 0.5
    beep_error
    print_ok "提示音测试完成"
fi

LAST_ACTION_ID=""
LAST_ORIENTATION=""
LAST_ENVIRONMENT_STATUS="clean"
LAST_DISTURBANCE_TAG="none"
LAST_FALL_DIRECTION="none"
LAST_HUMAN_ACTION_WINDOW_SEC="$DEFAULT_ACTION_WINDOW_SEC"
LAST_IDLE_ACTION_WINDOW_SEC=60
DRY_RUN_PREFIX_HISTORY=""

# ============================================================
# 8. 样本采集循环
# ============================================================

while true; do
    print_header
    echo "  新样本采集"
    echo "  输入 q 退出"
    echo ""

    ACTION_OPTIONS_TEXT=$(actions_for_environment_text "$ENVIRONMENT_ID")
    if [[ -n "$LAST_ACTION_ID" ]] && validate_action_for_environment "$ENVIRONMENT_ID" "$LAST_ACTION_ID"; then
        DEFAULT_ACTION_ID="$LAST_ACTION_ID"
    else
        DEFAULT_ACTION_ID=$(first_action_for_environment "$ENVIRONMENT_ID")
    fi

    # 输入 action_id（按当前 environment_id 验证）
    while true; do
        print_info "允许的 action_id [${ENVIRONMENT_ID}]: $ACTION_OPTIONS_TEXT"
        read_input "  🏷  action_id [${DEFAULT_ACTION_ID}/q]: " ACTION_ID
        ACTION_ID=${ACTION_ID:-$DEFAULT_ACTION_ID}
        ACTION_ID=$(echo "$ACTION_ID" | tr 'A-Z' 'a-z')

        if [[ "$ACTION_ID" == "q" || "$ACTION_ID" == "quit" || "$ACTION_ID" == "exit" ]]; then
            print_ok "退出采集"
            break 2
        fi

        if [[ -z "$ACTION_ID" ]]; then
            print_warn "action_id 不能为空"
            continue
        fi

        if validate_action_for_environment "$ENVIRONMENT_ID" "$ACTION_ID"; then
            LAST_ACTION_ID="$ACTION_ID"
            break
        else
            print_err "非法 action_id: $ACTION_ID"
            print_info "当前 environment_id=${ENVIRONMENT_ID} 允许的 action_id: $ACTION_OPTIONS_TEXT"
            beep_error
        fi
    done

    # 输入 orientation（带验证）
    DEFAULT_ORIENTATION="${LAST_ORIENTATION:-$(default_orientation_for_environment "$ENVIRONMENT_ID")}"
    while true; do
        read_input "  🧭 orientation [face_rx/face_tx/side_rx/side_link/cross_link] [${DEFAULT_ORIENTATION}]: " ORIENTATION
        ORIENTATION=${ORIENTATION:-$DEFAULT_ORIENTATION}
        ORIENTATION=$(echo "$ORIENTATION" | tr 'A-Z' 'a-z')

        if validate_orientation "$ORIENTATION"; then
            LAST_ORIENTATION="$ORIENTATION"
            break
        else
            print_err "非法 orientation: $ORIENTATION"
            print_info "允许的 orientation: ${VALID_ORIENTATIONS[*]}"
            beep_error
        fi
    done

    # 输入 environment_status（带验证）
    DEFAULT_ENVIRONMENT_STATUS="$LAST_ENVIRONMENT_STATUS"
    ENVIRONMENT_STATUS_OPTIONS=$(join_by_slash "${VALID_ENVIRONMENT_STATUS[@]}")
    while true; do
        read_input "  🎬 environment_status [${ENVIRONMENT_STATUS_OPTIONS}] [${DEFAULT_ENVIRONMENT_STATUS}]: " ENVIRONMENT_STATUS
        ENVIRONMENT_STATUS=${ENVIRONMENT_STATUS:-$DEFAULT_ENVIRONMENT_STATUS}
        ENVIRONMENT_STATUS=$(echo "$ENVIRONMENT_STATUS" | tr 'A-Z' 'a-z')

        if validate_environment_status "$ENVIRONMENT_STATUS"; then
            LAST_ENVIRONMENT_STATUS="$ENVIRONMENT_STATUS"
            break
        else
            print_err "非法 environment_status: $ENVIRONMENT_STATUS"
            print_info "允许的 environment_status: ${VALID_ENVIRONMENT_STATUS[*]}"
            beep_error
        fi
    done

    # 输入 disturbance_tag（带验证，默认 none）
    DEFAULT_DISTURBANCE_TAG="$LAST_DISTURBANCE_TAG"
    DISTURBANCE_TAG_OPTIONS=$(join_by_slash "${VALID_DISTURBANCE_TAG[@]}")
    while true; do
        read_input "  🌡  disturbance_tag [${DISTURBANCE_TAG_OPTIONS}] [${DEFAULT_DISTURBANCE_TAG}]: " DISTURBANCE_TAG
        DISTURBANCE_TAG=${DISTURBANCE_TAG:-$DEFAULT_DISTURBANCE_TAG}
        DISTURBANCE_TAG=$(echo "$DISTURBANCE_TAG" | tr 'A-Z' 'a-z')

        if validate_disturbance_tag "$DISTURBANCE_TAG"; then
            LAST_DISTURBANCE_TAG="$DISTURBANCE_TAG"
            break
        else
            print_err "非法 disturbance_tag: $DISTURBANCE_TAG"
            print_info "允许的 disturbance_tag: ${VALID_DISTURBANCE_TAG[*]}"
            beep_error
        fi
    done

    # fall_like / bed_fall_sim 可记录方向，非跌倒动作默认 none
    if [[ "$ACTION_ID" == "fall_like" || "$ACTION_ID" == "bed_fall_sim" ]]; then
        DEFAULT_FALL_DIRECTION="$LAST_FALL_DIRECTION"
        while true; do
            read_input "  ↘  fall_direction [none/forward/backward/left/right] [${DEFAULT_FALL_DIRECTION}]: " FALL_DIRECTION
            FALL_DIRECTION=${FALL_DIRECTION:-$DEFAULT_FALL_DIRECTION}
            FALL_DIRECTION=$(echo "$FALL_DIRECTION" | tr 'A-Z' 'a-z')

            if validate_fall_direction "$FALL_DIRECTION"; then
                LAST_FALL_DIRECTION="$FALL_DIRECTION"
                break
            else
                print_err "非法 fall_direction: $FALL_DIRECTION"
                print_info "允许的 fall_direction: ${VALID_FALL_DIRECTIONS[*]}"
                beep_error
            fi
        done
    else
        FALL_DIRECTION="none"
        print_info "非跌倒动作，fall_direction 自动设为 none"
    fi

    # 构建 sample_id 前缀
    PREFIX="${SESSION_ID}_${PERSON_ID}_${ACTION_ID}_${ORIENTATION}_${ENVIRONMENT_ID}_${ENVIRONMENT_STATUS}_${DISTURBANCE_TAG}"

    # 自动生成 trial_id
    if [[ "$DRY_RUN" -eq 1 ]]; then
        AUTO_TRIAL_NUM=$(dry_run_next_trial_number "$PREFIX")
    else
        AUTO_TRIAL_NUM=$(next_trial_number "$CSI_DIR" "$PREFIX")
    fi
    AUTO_TRIAL_ID=$(printf "R%03d" "$AUTO_TRIAL_NUM")

    read_input "  🔁 trial_id [${AUTO_TRIAL_ID}]: " TRIAL_ID
    TRIAL_ID=${TRIAL_ID:-$AUTO_TRIAL_ID}

    if [[ "$TRIAL_ID" =~ ^[0-9]+$ ]]; then
        TRIAL_ID=$(printf "R%03d" "$TRIAL_ID")
    fi

    TRIAL_ID=$(sanitize_field "$TRIAL_ID")

    if [[ "$PERSON_ID" == "NONE" || "$ACTION_ID" == "background_idle" ]]; then
        DEFAULT_ACTION_WINDOW_FOR_THIS_SAMPLE="$LAST_IDLE_ACTION_WINDOW_SEC"
    else
        DEFAULT_ACTION_WINDOW_FOR_THIS_SAMPLE="$LAST_HUMAN_ACTION_WINDOW_SEC"
    fi

    read_input "  ⏱  action_window_sec [${DEFAULT_ACTION_WINDOW_FOR_THIS_SAMPLE}]: " ACTION_WINDOW_SEC
    ACTION_WINDOW_SEC=${ACTION_WINDOW_SEC:-$DEFAULT_ACTION_WINDOW_FOR_THIS_SAMPLE}

    if ! [[ "$ACTION_WINDOW_SEC" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        print_warn "action_window_sec 必须是数字"
        continue
    fi

    if [[ "$PERSON_ID" == "NONE" || "$ACTION_ID" == "background_idle" ]]; then
        LAST_IDLE_ACTION_WINDOW_SEC="$ACTION_WINDOW_SEC"
    else
        LAST_HUMAN_ACTION_WINDOW_SEC="$ACTION_WINDOW_SEC"
    fi

    read_input "  🗒  notes，可留空: " NOTES

    SAMPLE_ID="${PREFIX}_${TRIAL_ID}"
    START_TIME_MS=0
    END_TIME_MS=$(awk -v s="$ACTION_WINDOW_SEC" 'BEGIN { printf "%.0f", s * 1000 }')

    DAT_FILE="${CSI_DIR}/${SAMPLE_ID}.dat"
    LOG_FILE="${LOG_DIR}/${SAMPLE_ID}.log"
    MARKER_FILE="${LOG_DIR}/${SAMPLE_ID}_markers.log"
    if [[ "$DRY_RUN" -eq 0 ]]; then
        CURRENT_DAT_FILE="$DAT_FILE"
    else
        CURRENT_DAT_FILE=""
    fi

    # 防止覆盖旧数据
    if [[ "$DRY_RUN" -eq 0 && ( -e "$DAT_FILE" || -e "$LOG_FILE" || -e "$MARKER_FILE" ) ]]; then
        print_err "样本已存在，防止覆盖：$SAMPLE_ID"
        print_info "DAT: $DAT_FILE"
        print_info "LOG: $LOG_FILE"
        print_info "MARKER: $MARKER_FILE"
        CURRENT_DAT_FILE=""
        continue
    fi

    # 检查 samples.csv 中是否已有此 sample_id
    if [[ "$DRY_RUN" -eq 0 && -f "$SAMPLES_CSV" ]] && grep -q "\"${SAMPLE_ID}\"" "$SAMPLES_CSV" 2>/dev/null; then
        print_err "sample_id 已存在于 samples.csv：$SAMPLE_ID"
        CURRENT_DAT_FILE=""
        continue
    fi

    print_header
    echo "  即将采集样本："
    echo ""
    echo "  sample_id        : $SAMPLE_ID"
    echo "  session_id       : $SESSION_ID"
    echo "  person_id        : $PERSON_ID"
    echo "  environment_id   : $ENVIRONMENT_ID"
    echo "  action_id        : $ACTION_ID"
    echo "  orientation      : $ORIENTATION"
    echo "  environment_status : $ENVIRONMENT_STATUS"
    echo "  disturbance_tag  : $DISTURBANCE_TAG"
    echo "  fall_direction   : $FALL_DIRECTION"
    echo "  trial_id         : $TRIAL_ID"
    echo "  action_window_sec: ${ACTION_WINDOW_SEC}s"
    echo "  start_time_ms    : $START_TIME_MS"
    echo "  end_time_ms      : $END_TIME_MS"
    echo "  csi_mode         : $CSI_MODE"
    echo "  frequency        : $FREQ"
    echo "  channel_width    : $CHANNEL_WIDTH"
    echo "  format           : $FORMAT"
    echo "  mcs              : $MCS"
    echo "  spatial_streams  : $SPATIAL_STREAMS"
    echo "  dat              : $DAT_FILE"
    echo "  log              : $LOG_FILE"
    echo "  marker           : $MARKER_FILE"
    echo ""
    print_tx_reference_command
    echo ""

    # 显示动作 SOP
    print_action_sop "$ACTION_ID"

    # dry-run 模式：显示 samples.csv 字段并跳过实际采集
    if [[ "$DRY_RUN" -eq 1 ]]; then
        print_header
        print_info "DRY-RUN 模式：samples.csv 字段预览"
        echo ""
        echo "  sample_id           : $SAMPLE_ID"
        echo "  session_id          : $SESSION_ID"
        echo "  person_id           : $PERSON_ID"
        echo "  environment_id      : $ENVIRONMENT_ID"
        echo "  orientation         : $ORIENTATION"
        echo "  action_id           : $ACTION_ID"
        echo "  environment_status  : $ENVIRONMENT_STATUS"
        echo "  disturbance_tag     : $DISTURBANCE_TAG"
        echo "  fall_direction      : $FALL_DIRECTION"
        echo "  trial_id            : $TRIAL_ID"
        echo "  action_window_sec   : $ACTION_WINDOW_SEC"
        echo "  csi_path            : $DAT_FILE"
        echo "  video_path          : ${VIDEO_PATH:-}"
        echo "  log_path            : $MARKER_FILE"
        echo "  start_time_ms       : $START_TIME_MS"
        echo "  end_time_ms         : $END_TIME_MS"
        echo "  quality_flag        : (采集时生成)"
        echo "  notes               : ${NOTES:-}"
        echo ""
        print_info "以下为 collection_manifest_raw.csv 采集过程字段，不属于 samples.csv"
        echo ""
        echo "  date                : $DATE_READABLE"
        echo "  csi_mode            : $CSI_MODE"
        echo "  frequency           : $FREQ"
        echo "  channel_width       : $CHANNEL_WIDTH"
        echo "  format              : $FORMAT"
        echo "  mcs                 : $MCS"
        echo "  spatial_streams     : $SPATIAL_STREAMS"
        echo "  marker_path         : $MARKER_FILE"
        echo "  srt_path            : ${SRT_PATH:-}"
        echo "  marker_count        : (采集时生成)"
        echo "  marker_status       : (采集时生成)"
        echo "  sample_start_iso    : (采集时生成)"
        echo "  sample_end_iso      : (采集时生成)"
        echo "  action_start_iso    : (采集时生成)"
        echo "  action_end_iso      : (采集时生成)"
        echo "  frame_count         : (采集时生成)"
        echo "  avg_fps             : (采集时生成)"
        echo "  file_size_bytes     : (采集时生成)"
        echo "  status              : (采集时生成)"
        echo ""

        print_info "DRY-RUN 模式：测试提示音序列"
        beep_ready
        sleep 0.3
        beep_countdown 3
        sleep 0.3
        beep_countdown 2
        sleep 0.3
        beep_countdown 1
        sleep 0.3
        beep_start
        sleep 0.5
        beep_end
        print_ok "DRY-RUN 完成，未启动 FeitCSI"
        record_dry_run_prefix "$PREFIX"
        CURRENT_DAT_FILE=""
        continue
    fi

    read_input "  按 Enter 开始初始化采集，输入 s 跳过: " CONFIRM

    if [[ "$CONFIRM" == "s" || "$CONFIRM" == "S" ]]; then
        print_warn "已跳过样本：$SAMPLE_ID"
        CURRENT_DAT_FILE=""
        continue
    fi

    > "$LOG_FILE"

    print_info "启动 FeitCSI..."
    print_info "等待 Enabling CSI measurement..."

    sudo stdbuf -oL feitcsi \
        --mode "$CSI_MODE" \
        --frequency "$FREQ" \
        --channel-width "$CHANNEL_WIDTH" \
        --format "$FORMAT" \
        --mcs "$MCS" \
        --spatial-streams "$SPATIAL_STREAMS" \
        --output-file "$DAT_FILE" \
        -v > "$LOG_FILE" 2>&1 &

    if ! wait_for_ready "$LOG_FILE" "$READY_TIMEOUT" "$DAT_FILE"; then
        print_err "FeitCSI 初始化失败或超时"

        stop_feitcsi_by_dat "$DAT_FILE"

        FRAME_COUNT=$(grep -c "Subcarrier count" "$LOG_FILE" 2>/dev/null || echo 0)
        FILE_SIZE_BYTES=$(file_size_bytes "$DAT_FILE")
        AVG_FPS="0.0"

        append_collection_manifest_raw "$COLLECTION_RAW_CSV" \
            "$SAMPLE_ID" "$SESSION_ID" "$DATE_READABLE" "$PERSON_ID" "$ENVIRONMENT_ID" \
            "$ACTION_ID" "$ORIENTATION" "$ENVIRONMENT_STATUS" "$DISTURBANCE_TAG" "$FALL_DIRECTION" "$TRIAL_ID" \
            "$ACTION_WINDOW_SEC" "$CSI_MODE" "$FREQ" "$CHANNEL_WIDTH" "$FORMAT" "$MCS" "$SPATIAL_STREAMS" \
            "$DAT_FILE" "$LOG_FILE" "$MARKER_FILE" "0" "missing_marker_file" "$VIDEO_PATH" "$SRT_PATH" \
            "" "" "" "" "" "" "" "" \
            "$FRAME_COUNT" "$AVG_FPS" "$FILE_SIZE_BYTES" "init_failed" "bad" "$NOTES"

        print_warn "失败记录已写入 collection_manifest_raw.csv，未写入正式 samples.csv"

        CURRENT_DAT_FILE=""
        continue
    fi

    # 记录样本开始时间
    SAMPLE_START_ISO=$(now_iso)
    SAMPLE_START_NS=$(now_ns)

    {
        echo ""
        echo "MARKER_SAMPLE_START ${SAMPLE_START_ISO} ${SAMPLE_START_NS}"
        echo "MARKER_SAMPLE_ID ${SAMPLE_ID}"
        echo "MARKER_SESSION_ID ${SESSION_ID}"
        echo "MARKER_PERSON_ID ${PERSON_ID}"
        echo "MARKER_ENVIRONMENT_ID ${ENVIRONMENT_ID}"
        echo "MARKER_ACTION_ID ${ACTION_ID}"
        echo "MARKER_ORIENTATION ${ORIENTATION}"
        echo "MARKER_ENVIRONMENT_STATUS ${ENVIRONMENT_STATUS}"
        echo "MARKER_DISTURBANCE_TAG ${DISTURBANCE_TAG}"
        echo "MARKER_FALL_DIRECTION ${FALL_DIRECTION}"
        echo "MARKER_TRIAL_ID ${TRIAL_ID}"
        echo "MARKER_ACTION_WINDOW_SEC ${ACTION_WINDOW_SEC}"
        echo "MARKER_WINDOW_POLICY action_start_to_action_end"
        echo "MARKER_PREWAIT_USAGE not_for_default_training"
        echo "MARKER_POST_BUFFER_SEC 0"
        echo "MARKER_CSI_MODE ${CSI_MODE}"
        echo "MARKER_CSI_FREQUENCY ${FREQ}"
        echo "MARKER_CSI_CHANNEL_WIDTH ${CHANNEL_WIDTH}"
        echo "MARKER_CSI_FORMAT ${FORMAT}"
        echo "MARKER_CSI_MCS ${MCS}"
        echo "MARKER_CSI_SPATIAL_STREAMS ${SPATIAL_STREAMS}"
        echo ""
    } >> "$MARKER_FILE"

    # 准备提示音
    print_info "准备提示音..."
    BEEP_READY_ISO=$(now_iso)
    BEEP_READY_NS=$(now_ns)
    beep_ready
    {
        echo "MARKER_BEEP_READY ${BEEP_READY_ISO} ${BEEP_READY_NS}"
    } >> "$MARKER_FILE"

    # 倒计时 3、2、1
    print_info "倒计时 3..."
    COUNTDOWN_3_ISO=$(now_iso)
    COUNTDOWN_3_NS=$(now_ns)
    beep_countdown 3
    {
        echo "MARKER_COUNTDOWN_3 ${COUNTDOWN_3_ISO} ${COUNTDOWN_3_NS}"
    } >> "$MARKER_FILE"
    sleep 0.8

    print_info "倒计时 2..."
    COUNTDOWN_2_ISO=$(now_iso)
    COUNTDOWN_2_NS=$(now_ns)
    beep_countdown 2
    {
        echo "MARKER_COUNTDOWN_2 ${COUNTDOWN_2_ISO} ${COUNTDOWN_2_NS}"
    } >> "$MARKER_FILE"
    sleep 0.8

    print_info "倒计时 1..."
    COUNTDOWN_1_ISO=$(now_iso)
    COUNTDOWN_1_NS=$(now_ns)
    beep_countdown 1
    {
        echo "MARKER_COUNTDOWN_1 ${COUNTDOWN_1_ISO} ${COUNTDOWN_1_NS}"
    } >> "$MARKER_FILE"
    sleep 0.8

    # 开始动作提示音
    print_ok "开始动作！（${ACTION_WINDOW_SEC}s）"
    ACTION_START_ISO=$(now_iso)
    ACTION_START_NS=$(now_ns)
    {
        echo "MARKER_ACTION_START ${ACTION_START_ISO} ${ACTION_START_NS}"
    } >> "$MARKER_FILE"
    beep_start

    STAT_PID=$(start_frame_monitor "$LOG_FILE" "$ACTION_WINDOW_SEC" "$DAT_FILE")

    # 默认训练窗口：MARKER_ACTION_START 到 MARKER_ACTION_END
    sleep "$ACTION_WINDOW_SEC"

    # 结束动作提示音
    ACTION_END_ISO=$(now_iso)
    ACTION_END_NS=$(now_ns)
    {
        echo "MARKER_ACTION_END ${ACTION_END_ISO} ${ACTION_END_NS}"
    } >> "$MARKER_FILE"
    beep_end

    # 记录样本结束时间
    SAMPLE_END_ISO=$(now_iso)
    SAMPLE_END_NS=$(now_ns)

    {
        echo ""
        echo "MARKER_SAMPLE_END ${SAMPLE_END_ISO} ${SAMPLE_END_NS}"
        echo ""
    } >> "$MARKER_FILE"

    stop_feitcsi_by_dat "$DAT_FILE"

    if [[ -n "${STAT_PID:-}" ]]; then
        kill "$STAT_PID" 2>/dev/null || true
        wait "$STAT_PID" 2>/dev/null || true
    fi

    FRAME_COUNT=$(grep -c "Subcarrier count" "$LOG_FILE" 2>/dev/null || echo 0)
    FILE_SIZE_BYTES=$(file_size_bytes "$DAT_FILE")
    FILE_SIZE_HUMAN=$(human_size "$DAT_FILE")
    RECORD_DURATION_SEC=$(awk -v start="$SAMPLE_START_NS" -v end="$SAMPLE_END_NS" 'BEGIN { if (end > start) printf "%.3f", (end - start) / 1000000000; else printf "0.000" }')
    AVG_FPS=$(calc_avg_fps "$FRAME_COUNT" "$RECORD_DURATION_SEC")

    # marker 完整性检查
    MARKER_RESULT=$(check_marker_status "$MARKER_FILE")
    MARKER_COUNT=$(echo "$MARKER_RESULT" | cut -d'|' -f1)
    MARKER_STATUS=$(echo "$MARKER_RESULT" | cut -d'|' -f2)

    STATUS="ok"
    QUALITY_FLAG="ok"

    if [[ "$FRAME_COUNT" -lt 100 ]]; then
        STATUS="warn_low_frames"
        QUALITY_FLAG="uncertain"
    fi

    append_samples_csv "$SAMPLES_CSV" \
        "$SAMPLE_ID" "$SESSION_ID" "$PERSON_ID" "$ENVIRONMENT_ID" "$ORIENTATION" "$ACTION_ID" \
        "$ENVIRONMENT_STATUS" "$DISTURBANCE_TAG" "$FALL_DIRECTION" "$TRIAL_ID" \
        "$DAT_FILE" "$VIDEO_PATH" "$MARKER_FILE" "$START_TIME_MS" "$END_TIME_MS" "$QUALITY_FLAG" "$NOTES"

    append_collection_manifest_raw "$COLLECTION_RAW_CSV" \
        "$SAMPLE_ID" "$SESSION_ID" "$DATE_READABLE" "$PERSON_ID" "$ENVIRONMENT_ID" \
        "$ACTION_ID" "$ORIENTATION" "$ENVIRONMENT_STATUS" "$DISTURBANCE_TAG" "$FALL_DIRECTION" "$TRIAL_ID" \
        "$ACTION_WINDOW_SEC" "$CSI_MODE" "$FREQ" "$CHANNEL_WIDTH" "$FORMAT" "$MCS" "$SPATIAL_STREAMS" \
        "$DAT_FILE" "$LOG_FILE" "$MARKER_FILE" "$MARKER_COUNT" "$MARKER_STATUS" "$VIDEO_PATH" "$SRT_PATH" \
        "$SAMPLE_START_ISO" "$SAMPLE_END_ISO" "$SAMPLE_START_NS" "$SAMPLE_END_NS" \
        "$ACTION_START_ISO" "$ACTION_END_ISO" "$ACTION_START_NS" "$ACTION_END_NS" \
        "$FRAME_COUNT" "$AVG_FPS" "$FILE_SIZE_BYTES" "$STATUS" "$QUALITY_FLAG" "$NOTES"

    echo -e "\n"
    print_ok "采集完成，已追加 samples.csv 和 collection_manifest_raw.csv"
    echo -e "  \033[0;37m────────────────────────────────────────────\033[0m"
    printf   "  %-18s %s\n" "sample_id" "$SAMPLE_ID"
    printf   "  %-18s %s\n" "action_id" "$ACTION_ID"
    printf   "  %-18s %s\n" "orientation" "$ORIENTATION"
    printf   "  %-18s %s\n" "environment_status" "$ENVIRONMENT_STATUS"
    printf   "  %-18s %s\n" "disturbance_tag" "$DISTURBANCE_TAG"
    printf   "  %-18s %s\n" "fall_direction" "$FALL_DIRECTION"
    printf   "  %-18s %s\n" "trial_id" "$TRIAL_ID"
    printf   "  %-18s %s\n" "action_window_sec" "$ACTION_WINDOW_SEC"
    printf   "  %-18s %s\n" "start_time_ms" "$START_TIME_MS"
    printf   "  %-18s %s\n" "end_time_ms" "$END_TIME_MS"
    printf   "  %-18s %s\n" "status" "$STATUS"
    printf   "  %-18s %s\n" "quality_flag" "$QUALITY_FLAG"
    printf   "  %-18s %s\n" "record_duration" "${RECORD_DURATION_SEC}s"
    printf   "  %-18s %s\n" "file size" "$FILE_SIZE_HUMAN"
    printf   "  %-18s %s\n" "frames" "$FRAME_COUNT"
    printf   "  %-18s %s\n" "avg fps" "~${AVG_FPS} Hz"
    printf   "  %-18s %s\n" "start" "$SAMPLE_START_ISO"
    printf   "  %-18s %s\n" "end" "$SAMPLE_END_ISO"
    printf   "  %-18s %s\n" "action_start" "$ACTION_START_ISO"
    printf   "  %-18s %s\n" "action_end" "$ACTION_END_ISO"
    printf   "  %-18s %s\n" "dat" "$DAT_FILE"
    printf   "  %-18s %s\n" "log" "$LOG_FILE"
    printf   "  %-18s %s\n" "markers" "$MARKER_FILE"
    printf   "  %-18s %s\n" "marker_status" "$MARKER_STATUS"
    printf   "  %-18s %s\n" "samples.csv" "$SAMPLES_CSV"
    printf   "  %-18s %s\n" "raw manifest" "$COLLECTION_RAW_CSV"
    echo -e "  \033[0;37m────────────────────────────────────────────\033[0m"

    CURRENT_DAT_FILE=""
    STAT_PID=""

done

CURRENT_DAT_FILE=""
STAT_PID=""

print_header
print_ok "本次 session 采集结束"
print_info "实验目录: $EXP_DIR"
print_info "样本清单: $SAMPLES_CSV"
print_info "原始采集记录: $COLLECTION_RAW_CSV"
