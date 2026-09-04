"""
TncodeSolver 使用示例
"""
from tncode_solver import TncodeSolver, SelectorConfig


def solve_default(page):
    """默认选择器求解。"""
    solver = TncodeSolver(page, data_file="data.json")
    return solver.solve()


def solve_custom_site(page):
    """
    自定义选择器 — 适配不同 class/id 的 tncode 站点。
    """
    sel = SelectorConfig()
    sel.canvas_bg = "#captcha_bg"
    sel.canvas_mark = "#captcha_mask"
    sel.slide_block = ".captcha-slider"
    sel.refresh_btn = ".captcha-refresh"
    sel.close_btn = ".captcha-close"
    sel.tncode_global = "verifyCode"
    sel.tncode_div = ".captcha-wrapper"

    solver = TncodeSolver(page, data_file="custom_data.json", selectors=sel)
    return solver.solve()


def full_flow_example():
    """完整登录+验证码流程。"""
    from DrissionPage import ChromiumPage

    page = ChromiumPage()
    page.get("https://example.com/login")
    # ... 登录逻辑 ...

    solver = TncodeSolver(page, data_file="data.json")
    if solver.solve():
        print("验证码通过")
    else:
        print("验证码失败")


if __name__ == "__main__":
    print("用法: from tncode_solver import TncodeSolver, SelectorConfig")
