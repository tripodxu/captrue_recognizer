"""
TncodeSolver 使用示例

演示如何在不同站点中使用通用求解器。
"""

from tncode_solver import TncodeSolver, SelectorConfig

# ---------------------------------------------------------------------------
#  示例 自定义选择器 (适配其他使用 tncode 的站点)
# ---------------------------------------------------------------------------

def solve_custom_site(page):
    """
    假设某个站点使用了 tncode 但 class 名不同:
      - 背景 canvas:   #captcha_bg
      - 切片 canvas:   #captcha_mask
      - 滑块:          .captcha-slider
      - 刷新按钮:      .captcha-refresh
      - 关闭按钮:      .captcha-close
      - tncode 对象:   window.verifyCode
    """
    sel = SelectorConfig()
    sel.canvas_bg = "#captcha_bg"
    sel.canvas_mark = "#captcha_mask"
    sel.slide_block = ".captcha-slider"
    sel.refresh_btn = ".captcha-refresh"
    sel.close_btn = ".captcha-close"
    sel.tncode_global = "verifyCode"
    sel.tncode_div = ".captcha-wrapper"

    solver = TncodeSolver(
        page,
        data_file="custom_data.json",
        selectors=sel,
    )
    return solver.solve()


# ---------------------------------------------------------------------------
#  示例 2: DrissionPage 完整流程
# ---------------------------------------------------------------------------

def full_flow_example():
    """完整的登录+发帖验证码流程示例。"""
    from DrissionPage import ChromiumPage

    page = ChromiumPage()
    page.get("https://example.com/login")

    # ... 登录逻辑 ...

    # 检测到验证码时
    solver = TncodeSolver(page, data_file="data.json")
    if solver.solve():
        print("验证码通过，继续操作")
    else:
        print("验证码失败，请手动处理")


if __name__ == "__main__":
    print("这是一个库模块，请在你的项目中 import 使用。")
    print("用法: from tncode_solver import TncodeSolver, SelectorConfig")
