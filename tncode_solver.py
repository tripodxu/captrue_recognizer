"""
tncode 滑动验证码通用求解器 (通用版)


三层求解策略:
1. 感知哈希模糊查找 (data.json 已学习数据，秒级)
2. OpenCV CV 检测 (v5: 多阈值 Canny + NPC 置信度择优，自动学习)
3. 等待用户手动拖拽 → 记录学习

反检测 / 人机行为模拟:
- 随机鼠标游走
- 渐进式滑块触发 (含微抖动)
- 自然加减速轨迹 + 过冲回弹 + y 轴偏移
- CDP 原生鼠标事件 (比 Selenium 更难检测)
- 拖拽中随机暂停 (15% 概率)

卡住检测:
- 连续 2 次相同哈希 → 点击刷新
- 连续 3 次相同哈希 → 关闭重开

依赖: drissionpage, opencv-python, numpy

用法:
    from tncode_solver import TncodeSolver
    solver = TncodeSolver(page, data_file="data.json")
    ok = solver.solve()
"""

import json
import time
import random
import base64
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np


# ---------------------------------------------------------------------------
#  CSS / JS 选择器配置 (可按站点覆盖)
# ---------------------------------------------------------------------------

class SelectorConfig:
    """
    tncode 验证码 DOM 选择器配置。
    不同站点可能使用不同的 class / id，通过此类解耦。
    """

    # canvas 元素
    canvas_bg: str = ".tncode_canvas_bg"
    canvas_mark: str = ".tncode_canvas_mark"

    # 滑块
    slide_block: str = ".slide_block"

    # 刷新按钮
    refresh_btn: str = ".tncode_refresh"

    # 关闭按钮
    close_btn: str = ".tncode_close"

    # tncode 全局对象名
    tncode_global: str = "tncode"

    # 验证码容器 (触发 reopen 用)
    tncode_div: str = ".tncode"

    # 滑块成功 class 关键字
    success_keywords: tuple = ("ok", "success")


class TncodeSolver:
    """
    通用 tncode 滑动验证码求解器。

    参数:
        page: DrissionPage 的页面对象 (ChromiumPage / Tab)
        data_file: 哈希→距离学习数据的 JSON 文件路径
        selectors: CSS 选择器配置 (可选，默认 SelectorConfig)
        canvas_width: canvas 原始宽度 (用于 scale 计算兜底)
    """

    def __init__(self, page, data_file="data.json", selectors=None, canvas_width=300):
        self.page = page
        self.data_file = Path(data_file)
        self.sel = selectors or SelectorConfig()
        self.canvas_width = canvas_width
        self._data: Dict[str, int] = self._load_data()

    # ------------------------------------------------------------------
    #  数据持久化
    # ------------------------------------------------------------------

    def _load_data(self) -> dict:
        if self.data_file.exists():
            return json.loads(self.data_file.read_text(encoding="utf-8-sig"))
        return {}

    def _save_data(self):
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        self.data_file.write_text(json.dumps(self._data, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    #  人机行为模拟
    # ------------------------------------------------------------------

    def _human_delay(self, lo=0.3, hi=1.2):
        """随机延迟，模拟人类反应时间。"""
        time.sleep(random.uniform(lo, hi))

    def _random_mouse_wander(self, n=None):
        """交互前随机移动鼠标 n 次。"""
        if n is None:
            n = random.randint(1, 3)
        try:
            viewport = self.page.run_js("return {w:window.innerWidth,h:window.innerHeight};")
            vw, vh = viewport.get("w", 1200), viewport.get("h", 800)
            cx, cy = random.randint(100, vw - 100), random.randint(100, vh - 100)
            for _ in range(n):
                tx = cx + random.randint(-200, 200)
                ty = cy + random.randint(-150, 150)
                tx = max(50, min(vw - 50, tx))
                ty = max(50, min(vh - 50, ty))
                steps = random.randint(5, 12)
                for s in range(steps):
                    ix = cx + (tx - cx) * (s + 1) // steps + random.randint(-2, 2)
                    iy = cy + (ty - cy) * (s + 1) // steps + random.randint(-2, 2)
                    self._cdp("mouseMoved", ix, iy)
                    time.sleep(random.uniform(0.005, 0.02))
                cx, cy = tx, ty
        except Exception:
            pass

    # ------------------------------------------------------------------
    #  感知哈希
    # ------------------------------------------------------------------

    def _combo_hash(self, bg_img, mark_img) -> str:
        """背景+切片图拼接后计算均值二值化哈希。"""
        bg_s = cv2.resize(cv2.cvtColor(bg_img, cv2.COLOR_BGR2GRAY), (16, 8))
        mark_s = cv2.resize(cv2.cvtColor(mark_img, cv2.COLOR_BGR2GRAY), (16, 8))
        combined = np.vstack([bg_s, mark_s])
        avg = combined.mean()
        bits = 0
        for row in combined:
            for px in row:
                bits = (bits << 1) | (1 if px > avg else 0)
        return format(bits, f"0{(combined.size + 3) // 4}x")

    def _fuzzy_lookup(self, target_hash: str, max_distance=5) -> Optional[int]:
        """在已学习数据中做汉明距离模糊匹配。"""
        if not self._data:
            return None
        if target_hash in self._data:
            return self._data[target_hash]
        try:
            target_int = int(target_hash, 16)
        except ValueError:
            return None
        best_dist = max_distance + 1
        best_val = None
        for h, val in self._data.items():
            try:
                dist = bin(int(h, 16) ^ target_int).count("1")
                if dist < best_dist:
                    best_dist = dist
                    best_val = val
            except ValueError:
                continue
        return best_val if best_dist <= max_distance else None

    # ------------------------------------------------------------------
    #  CV 检测 (v5: 多阈值 Canny + NPC 置信度择优)
    # ------------------------------------------------------------------

    _CANNY_THRESHOLDS = [(30, 100), (50, 150), (80, 180), (100, 200),
                         (120, 240), (150, 250), (180, 300)]

    def _detect_cv(self, bg_img, mark_img) -> Tuple[Optional[int], float]:
        """
        v5 CV 检测 — 多阈值 Canny + NPC 置信度择优，无 SSD。

        策略:
        1. 多阈值 Canny 边缘检测 (7 组阈值)，取 CCOEFF_NORMED 最高者 → MC 结果。
        2. NPC 单阈值 (150,250) 单独匹配 → NPC 结果。
        3. 若 MC 置信度 >= 0.35 或两者 x 差距 <= 5px → 信任 MC。
        4. 否则取 MC 和 NPC 中置信度更高者。

        返回 (x_offset, confidence)。
        """
        try:
            if mark_img is None:
                return None, 0

            bg = cv2.normalize(bg_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            pz = cv2.normalize(mark_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
            pz_gray = cv2.cvtColor(pz, cv2.COLOR_BGR2GRAY)

            # 多阈值 Canny
            best_val = -1.0
            best_x = 0
            for lo, hi in self._CANNY_THRESHOLDS:
                bg_e = cv2.Canny(bg_gray, lo, hi)
                pz_e = cv2.Canny(pz_gray, lo, hi)
                result = cv2.matchTemplate(bg_e, pz_e, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                if max_val > best_val:
                    best_val = max_val
                    best_x = max_loc[0]

            # NPC 一致性校验
            bg_npc = cv2.Canny(bg_gray, 150, 250)
            pz_npc = cv2.Canny(pz_gray, 150, 250)
            npc_result = cv2.matchTemplate(bg_npc, pz_npc, cv2.TM_CCOEFF_NORMED)
            npc_val, _, _, npc_loc = cv2.minMaxLoc(npc_result)

            mc_conf = max(0.0, min(1.0, best_val))
            npc_conf = max(0.0, min(1.0, npc_val))

            if mc_conf >= 0.35 or abs(best_x - npc_loc[0]) <= 5:
                print(f"    [CV-multi] x={best_x} conf={mc_conf:.3f}")
                return best_x, mc_conf

            # 置信度择优
            if npc_conf > mc_conf:
                print(f"    [CV-npc] x={npc_loc[0]} conf={npc_conf:.3f} (mc_conf={mc_conf:.3f})")
                return npc_loc[0], npc_conf

            print(f"    [CV-multi] x={best_x} conf={mc_conf:.3f}")
            return best_x, mc_conf

        except Exception as e:
            print(f"    [CV error] {e}")
            return None, 0

    # ------------------------------------------------------------------
    #  等待用户手动拖拽
    # ------------------------------------------------------------------

    def _watch_user_drag(self, timeout=90) -> Optional[int]:
        """等待用户手动拖拽滑块，返回拖拽距离 (canvas 坐标)。"""
        sel = self.sel
        for _ in range(timeout * 2):
            result = self.page.run_js(f"""
                if(typeof {sel.tncode_global}!=='undefined'){{
                    if({sel.tncode_global}.result===true||{sel.tncode_global}._result===true)return 'OK';
                }}
                var b=document.querySelector('{sel.slide_block}');
                if(b){{var c=b.className||'';if(c.includes('ok')||c.includes('success'))return 'OK';}}
                return 'PENDING';
            """)
            if result == "OK":
                pos = self.page.run_js(f"""
                    var b=document.querySelector('{sel.slide_block}');
                    var g=document.querySelector('{sel.canvas_bg}');
                    if(!b||!g)return null;
                    return {{bx:b.getBoundingClientRect().x,gx:g.getBoundingClientRect().x,gw:g.getBoundingClientRect().width,cw:g.width}};
                """)
                if pos:
                    s = pos["gw"] / pos["cw"] if pos.get("cw", 0) > 0 else 1.0
                    return int((pos["bx"] - pos["gx"]) / s) if s > 0 else 0
                return None
            time.sleep(0.5)
        return None

    # ------------------------------------------------------------------
    #  鼠标拖拽 (CDP)
    # ------------------------------------------------------------------

    def _cdp(self, event_type: str, x, y):
        """通过 CDP 发送鼠标事件。"""
        params = {"type": event_type, "x": round(x), "y": round(y)}
        if event_type != "mouseMoved":
            params["button"] = "left"
            params["clickCount"] = 1
        self.page.run_cdp("Input.dispatchMouseEvent", **params)

    def _trigger_gap(self) -> int:
        """
        触发滑块：移动到滑块附近 → 按下 → 渐进拖动 (含微抖动 + 随机暂停)。
        返回触发阶段已移动的像素距离。
        """
        sel = self.sel
        try:
            pos = self.page.run_js(f"""
                var b=document.querySelector('{sel.slide_block}');
                if(!b)return null;
                var r=b.getBoundingClientRect();
                return {{x:r.x+r.width/2,y:r.y+r.height/2}};
            """)
            if not pos:
                return 0
            sx, sy = pos["x"], pos["y"]

            # 随机靠近
            pre_x = sx + random.uniform(-15, 15)
            pre_y = sy + random.uniform(-8, 8)
            self._cdp("mouseMoved", pre_x, pre_y)
            time.sleep(random.uniform(0.1, 0.3))

            # 精确移动到滑块
            self._cdp("mouseMoved", sx, sy)
            time.sleep(random.uniform(0.08, 0.2))

            # 按下
            self._cdp("mousePressed", sx, sy)
            time.sleep(random.uniform(0.15, 0.35))

            # 渐进拖动 + 微抖动 + 随机暂停
            cx, cy = sx, sy
            steps = random.randint(8, 15)
            for _ in range(steps):
                dx = random.uniform(1, 3)
                dy = random.uniform(-0.8, 0.8)
                cx += dx
                cy += dy
                self._cdp("mouseMoved", cx, cy)
                dt = random.uniform(0.01, 0.04)
                if random.random() < 0.15:
                    dt += random.uniform(0.05, 0.15)
                time.sleep(dt)

            time.sleep(random.uniform(0.2, 0.5))
            return cx - sx
        except Exception as e:
            print(f"    [trigger error] {e}")
            return 0

    def _release_slider(self):
        """释放鼠标左键。"""
        try:
            self._cdp("mouseReleased", 0, 0)
        except Exception:
            pass

    def _auto_drag(self, distance: float, already_held=False, offset=0) -> bool:
        """自动拖拽滑块到目标距离。"""
        sel = self.sel
        try:
            pos = self.page.run_js(f"""
                var b=document.querySelector('{sel.slide_block}');
                if(!b)return null;
                var r=b.getBoundingClientRect();
                return {{x:r.x+r.width/2,y:r.y+r.height/2}};
            """)
            if not pos:
                return False
            adjusted = max(0, distance - offset)
            track = self._build_track(adjusted)
            sx, sy = pos["x"], pos["y"]
            if not already_held:
                self._cdp("mousePressed", sx, sy)
                time.sleep(random.uniform(0.05, 0.15))
            cx, cy = sx, sy
            for dx, dy, dt in track:
                cx += dx
                cy += dy
                self._cdp("mouseMoved", cx, cy)
                time.sleep(dt)
            time.sleep(random.uniform(0.08, 0.2))
            self._cdp("mouseReleased", cx, cy)
            return True
        except Exception as e:
            print(f"    [drag error] {e}")
            return False

    def _build_track(self, distance: float):
        """
        构建自然拖拽轨迹：前段加速 + 后段减速 + 过冲回弹 + y 轴微偏移。
        返回 [(dx, dy, dt), ...]
        """
        track = []
        overshoot = random.uniform(3, 8)
        target = distance + overshoot
        cur, mid = 0, target * 0.7
        v, t = 0, random.uniform(0.15, 0.22)
        while cur < target:
            a = random.uniform(5, 8) if cur < mid else random.uniform(-5, -2)
            v = max(v + a * t, 0.3)
            move = v * t + 0.5 * a * t * t
            if abs(move) < 0.5:
                move = 0.5
            cur += abs(move)
            dt = random.uniform(0.005, 0.015) if cur < mid else random.uniform(0.01, 0.04)
            if random.random() < 0.05:
                dt += random.uniform(0.03, 0.08)
            dy = random.choice([-1, 0, 0, 0, 1])
            track.append((round(abs(move)), dy, dt))
        # 过冲回弹
        back = overshoot + random.uniform(0.5, 2)
        steps = random.randint(3, 6)
        per = back / steps
        for _ in range(steps):
            track.append((-round(per), random.randint(-1, 1), random.uniform(0.02, 0.06)))
        # 精确修正
        total = sum(s[0] for s in track)
        diff = round(distance) - total
        if diff:
            track.append((diff, 0, 0.03))
        return track

    # ------------------------------------------------------------------
    #  DOM 基础操作
    # ------------------------------------------------------------------

    def _get_images(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int, bool]:
        """
        获取背景图和切片图，最多重试 3 次。
        返回 (bg, mark, trigger_offset, slider_held)。
        """
        sel = self.sel
        for try_i in range(3):
            offset = self._trigger_gap()
            self.page.run_js(f"""
                try{{
                    var bg=document.querySelector('{sel.canvas_bg}');
                    var mk=document.querySelector('{sel.canvas_mark}');
                    if(bg)bg.style.display='block';
                    if(mk)mk.style.display='block';
                }}catch(e){{}}
            """)
            time.sleep(0.3)

            bg_b64 = self.page.run_js(
                f"var c=document.querySelector('{sel.canvas_bg}');"
                "return c?c.toDataURL('image/png').split(',')[1]:null;"
            )
            self.page.run_js(f"""
                try{{
                    if(typeof {sel.tncode_global}!=='undefined'){{
                        try{{{sel.tncode_global}._draw_mark();}}catch(e){{}}
                    }}
                }}catch(e){{}}
            """)
            time.sleep(0.2)
            mark_b64 = self.page.run_js(
                f"var c=document.querySelector('{sel.canvas_mark}');"
                "return c?c.toDataURL('image/png').split(',')[1]:null;"
            )

            bg = cv2.imdecode(np.frombuffer(base64.b64decode(bg_b64), np.uint8), cv2.IMREAD_COLOR) if bg_b64 else None
            mark = cv2.imdecode(np.frombuffer(base64.b64decode(mark_b64), np.uint8), cv2.IMREAD_COLOR) if mark_b64 else None

            if bg is not None and mark is not None:
                nz = np.count_nonzero(cv2.cvtColor(mark, cv2.COLOR_BGR2GRAY))
                if nz >= 100:
                    return bg, mark, offset, True
                print(f"    [retry-img] mark too sparse (nz={nz}), attempt {try_i+1}/3")
            else:
                print(f"    [retry-img] bg/mark is None, attempt {try_i+1}/3")

            self._release_slider()
            self._human_delay(0.5, 1.0)
            self._refresh()

        return bg, mark, offset, False

    def _wait_ready(self, timeout=10) -> bool:
        """等待 canvas 渲染完成。"""
        sel = self.sel
        for _ in range(timeout * 2):
            ok = self.page.run_js(f"""
                try{{
                    var c=document.querySelector('{sel.canvas_bg}');
                    if(!c)return false;
                    var d=c.getContext('2d').getImageData(0,0,c.width,c.height).data;
                    var n=0;
                    for(var i=0;i<d.length;i+=16){{if(d[i]>0||d[i+1]>0||d[i+2]>0)n++;}}
                    return n>100;
                }}catch(e){{return false;}}
            """)
            if ok:
                return True
            time.sleep(0.5)
        return False

    def _refresh(self):
        """刷新验证码。"""
        sel = self.sel
        try:
            self.page.run_js(f"""
                if(typeof {sel.tncode_global}!=='undefined'&&{sel.tncode_global}.refresh){sel.tncode_global}.refresh();
                else{{var b=document.querySelector('{sel.refresh_btn}');if(b)b.click();}}
            """)
        except Exception:
            pass
        self._human_delay(1.5, 3.0)

    def _click_refresh_button(self):
        """精确点击刷新按钮 (CDP 模拟鼠标)。"""
        sel = self.sel
        try:
            pos = self.page.run_js(f"""
                var b=document.querySelector('{sel.refresh_btn}');
                if(!b)return null;
                var r=b.getBoundingClientRect();
                return {{x:r.x+r.width/2,y:r.y+r.height/2}};
            """)
            if pos:
                pre_x = pos["x"] + random.uniform(-10, 10)
                pre_y = pos["y"] + random.uniform(-10, 10)
                self._cdp("mouseMoved", pre_x, pre_y)
                self._human_delay(0.2, 0.5)
                self._cdp("mouseMoved", pos["x"], pos["y"])
                self._human_delay(0.1, 0.3)
                self._cdp("mousePressed", pos["x"], pos["y"])
                self._human_delay(0.05, 0.12)
                self._cdp("mouseReleased", pos["x"], pos["y"])
            else:
                self.page.run_js(f"var b=document.querySelector('{sel.refresh_btn}');if(b)b.click();")
        except Exception:
            try:
                self.page.run_js(f"var b=document.querySelector('{sel.refresh_btn}');if(b)b.click();")
            except Exception:
                pass
        self._human_delay(1.5, 3.0)

    def _reopen_post(self):
        """关闭并重新打开验证码。"""
        sel = self.sel
        try:
            self.page.run_js(f"document.querySelector('{sel.tncode_div}')?.click();")
        except Exception:
            pass
        self._human_delay(1.5, 3.0)

    def _get_scale(self) -> float:
        """计算 canvas 到显示的缩放比例。"""
        sel = self.sel
        try:
            info = self.page.run_js(f"""
                var bg=document.querySelector('{sel.canvas_bg}');
                if(!bg)return null;
                var r=bg.getBoundingClientRect();
                var dw=r.width||0;
                if(dw<10){{var d=document.querySelector('.tncode_div');if(d){{var dr=d.getBoundingClientRect();dw=dr.width-20;}}}}
                return {{cw:bg.width,dw:dw}};
            """)
            if not info:
                return 1.0
            cw = info.get("cw", self.canvas_width) or self.canvas_width
            dw = info.get("dw", 0) or 0
            return (dw / cw) if cw > 0 and dw > 10 else 1.0
        except Exception:
            return 1.0

    def _check_success(self) -> bool:
        """检查验证码是否通过。"""
        sel = self.sel
        result = self.page.run_js(f"""
            if(typeof {sel.tncode_global}!=='undefined'){{
                if({sel.tncode_global}.result===true||{sel.tncode_global}._result===true)return 'SUCCESS';
            }}
            var b=document.querySelector('{sel.slide_block}');
            if(b){{var c=b.className||'';if(c.includes('ok')||c.includes('success'))return 'SUCCESS';}}
            return 'UNKNOWN';
        """)
        return result == "SUCCESS"

    # ------------------------------------------------------------------
    #  主流程
    # ------------------------------------------------------------------

    def solve(self, max_retries=8) -> bool:
        """
        主求解流程，最多尝试 max_retries 次。
        返回 True 表示验证码通过。
        """
        last_h = None
        stuck = 0
        slider_held = False
        trigger_offset = 0

        for attempt in range(1, max_retries + 1):
            if attempt > 1:
                print(f"    [retry] {attempt}/{max_retries}...")
                if slider_held:
                    self._release_slider()
                    slider_held = False
                self._human_delay(0.8, 2.0)
                self._refresh()

            if not self._wait_ready():
                print("    [error] canvas timeout")
                continue

            # 40% 概率随机游走鼠标
            if random.random() < 0.4:
                self._random_mouse_wander(random.randint(1, 2))
                self._human_delay(0.3, 0.8)

            bg, mark, offset, held = self._get_images()
            slider_held = held
            trigger_offset = offset
            if bg is None or mark is None:
                print("    [error] no bg/mark")
                if slider_held:
                    self._release_slider()
                    slider_held = False
                self._refresh()
                continue

            h = self._combo_hash(bg, mark) if mark is not None else ""

            # 卡住检测
            if h and h == last_h:
                stuck += 1
                if stuck == 2:
                    print("    [stuck] same captcha 2x, clicking refresh...")
                    self._release_slider()
                    slider_held = False
                    self._click_refresh_button()
                    self._human_delay(1.5, 3.0)
                    continue
                if stuck >= 3:
                    print("    [stuck] same captcha 3x, re-entering...")
                    self._release_slider()
                    slider_held = False
                    try:
                        sel = self.sel
                        self.page.run_js(
                            f"var c=document.querySelector('{sel.close_btn}');"
                            "if(c)c.click();"
                        )
                    except Exception:
                        pass
                    self._human_delay(1.0, 2.0)
                    self._reopen_post()
                    stuck = 0
                    continue
            else:
                stuck = 0
            last_h = h
            distance = None

            # 策略 1: 哈希查找
            match = self._fuzzy_lookup(h)
            if match is not None:
                distance = match
                print(f"    [lookup] hash={h[:8]}... -> x={distance}")

            # 策略 2: CV 检测
            if distance is None:
                distance, conf = self._detect_cv(bg, mark)
                if distance is not None:
                    if h and h not in self._data:
                        self._data[h] = distance
                        self._save_data()
                        print(f"    [saved] CV result hash={h[:8]}... -> x={distance}")

            # 策略 3: 等待用户
            if distance is None:
                print(f"    [lookup] hash={h[:8]}..., CV failed")
                print("    >>> DRAG SLIDER MANUALLY <<<")
                x = self._watch_user_drag(timeout=90)
                if x is not None:
                    if h:
                        self._data[h] = x
                        self._save_data()
                        print(f"    [learned] hash={h[:8]}... -> x={x}")
                    distance = x

            if distance is None or distance <= 0:
                print("    [error] all methods failed")
                self._release_slider()
                slider_held = False
                continue

            # 执行拖拽
            self._human_delay(0.4, 1.2)
            scale = self._get_scale()
            real_distance = distance * scale
            print(f"    [drag] x={distance} * {scale:.2f} = {real_distance:.0f}px")

            if self._auto_drag(real_distance, already_held=slider_held, offset=trigger_offset):
                slider_held = False
                self._human_delay(1.0, 2.5)
                if self._check_success():
                    print("    [OK] captcha passed!")
                    return True
                self._human_delay(0.5, 1.0)

        print("    [FAIL] all attempts failed")
        return False
