"""Measure MiniMind's tool use on eight fixed synthetic order questions."""

import argparse
from decimal import Decimal
import json
from pathlib import Path
import re


CASES = [
    ("A102", "P241", 3, "19.90", "A102 这笔订单一共要付多少钱？"),
    ("A157", "P305", 2, "45.50", "帮我算一下订单 A157 的商品总金额。"),
    ("B203", "P417", 5, "8.75", "请核对 B203 这单的商品合计。"),
    ("B286", "P539", 7, "12.40", "订单 B286 的商品加起来多少钱？"),
    ("C319", "P623", 1, "88.88", "我想知道 C319 这张订单的应付金额。"),
    ("C374", "P758", 4, "6.25", "查一下 C374 这单的商品总价。"),
    ("D428", "P819", 6, "23.99", "请问订单 D428 一共多少钱？"),
    ("D581", "P946", 9, "3.20", "帮我确认 D581 的应付总额。"),
]
ORDERS = {
    order_id: {"product_id": product_id, "quantity": quantity}
    for order_id, product_id, quantity, _, _ in CASES
}
PRICES = {product_id: price for _, product_id, _, price, _ in CASES}


def tool(name, description, properties):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
            },
        },
    }


TOOLS = [
    tool("lookup_order", "按订单编号查询商品编号和购买数量。", {
        "order_id": {"type": "string", "description": "用户提供的订单编号"},
    }),
    tool("lookup_price", "按商品编号查询商品单价，单位为元。", {
        "product_id": {"type": "string", "description": "订单返回的商品编号"},
    }),
    tool("calculate_total", "用购买数量乘以商品单价，计算总金额。", {
        "quantity": {"type": "integer", "description": "购买数量"},
        "unit_price": {"type": "string", "description": "商品单价，单位为元"},
    }),
]
TOOL_ORDER = tuple(entry["function"]["name"] for entry in TOOLS)
MAX_TURNS = 6


def money(value):
    text = str(value)
    if re.fullmatch(r"[0-9]{1,6}(?:\.[0-9]{1,2})?", text):
        return Decimal(text)
    return None


def execute_tool(name, arguments):
    if not isinstance(arguments, dict):
        return {"error": "工具参数必须是 JSON 对象"}
    if name == "lookup_order":
        order_id = arguments.get("order_id")
        if isinstance(order_id, str) and order_id in ORDERS:
            return ORDERS[order_id].copy()
        return {"error": "找不到该订单"}
    if name == "lookup_price":
        product_id = arguments.get("product_id")
        if isinstance(product_id, str) and product_id in PRICES:
            return {"unit_price": PRICES[product_id]}
        return {"error": "找不到该商品"}
    if name == "calculate_total":
        quantity = arguments.get("quantity")
        price = money(arguments.get("unit_price"))
        if type(quantity) is int and 1 <= quantity <= 999 and price is not None:
            return {"total": f"{quantity * price:.2f}"}
        return {"error": "数量或单价格式错误"}
    return {"error": "未知工具"}


def run_case(case, generate, parse_calls):
    order_id, product_id, quantity, price, question = case
    messages = [{
        "role": "user",
        "content": question + ' 请使用工具核实数据。最终只输出 JSON，例如 {"total":"0.00"}，单位为元。',
    }]
    calls = []
    answer = ""
    for turn in range(1, MAX_TURNS + 1):
        response = generate(messages)
        generated_calls = parse_calls(response)
        if not generated_calls:
            answer = response
            break
        messages.append({"role": "assistant", "content": response})
        for generated_call in generated_calls:
            entry = generated_call if isinstance(generated_call, dict) else {}
            name = entry.get("name")
            arguments = entry.get("arguments")
            result = execute_tool(name, arguments)
            calls.append({"turn": turn, "name": name, "arguments": arguments, "result": result})
            messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)})

    sequence_ok = (
        tuple(call["name"] for call in calls) == TOOL_ORDER
        and [call["turn"] for call in calls] == [1, 2, 3]
    )
    arguments_ok = False
    if sequence_ok:
        order_args, price_args, calc_args = [call["arguments"] for call in calls]
        arguments_ok = (
            all("error" not in call["result"] for call in calls)
            and order_args.get("order_id") == order_id
            and price_args.get("product_id") == product_id
            and calc_args.get("quantity") == quantity
            and money(calc_args.get("unit_price")) == Decimal(price)
        )

    expected_total = quantity * Decimal(price)
    try:
        final = json.loads(answer.rsplit("</think>", 1)[-1].strip())
    except json.JSONDecodeError:
        final = None
    amount_ok = isinstance(final, dict) and money(final.get("total")) == expected_total
    return {
        "order_id": order_id,
        "question": question,
        "expected_total": f"{expected_total:.2f}",
        "calls": calls,
        "answer": answer,
        "sequence_ok": sequence_ok,
        "arguments_ok": arguments_ok,
        "amount_ok": amount_ok,
        "success": arguments_ok and amount_ok,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight", default="full_sft", help="Prefix of a native checkpoint in out/")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from eval_toolcall import generate, init_model, parse_tool_calls, setup_seed

    args.load_from = "./model"
    args.save_dir = "out"
    args.hidden_size = 768
    args.num_hidden_layers = 8
    args.use_moe = 0
    args.max_new_tokens = 128
    args.temperature = 0.7
    args.top_p = 0.9
    args.show_speed = False
    model, tokenizer = init_model(args)

    results = []
    for index, case in enumerate(CASES):
        setup_seed(args.seed + index)
        print(f"\n[Order {case[0]}] {case[-1]}", flush=True)
        result = run_case(
            case,
            lambda messages: generate(model, tokenizer, messages, TOOLS, args),
            parse_tool_calls,
        )
        results.append(result)
        print(
            f"order={case[0]} sequence={result['sequence_ok']} "
            f"arguments={result['arguments_ok']} amount={result['amount_ok']} "
            f"success={result['success']}"
        )

    summary = {
        key: sum(result[key] for result in results)
        for key in ("sequence_ok", "arguments_ok", "amount_ok", "success")
    }
    path = Path("out") / f"order_eval_{args.weight}.json"
    report = {"weight": args.weight, "seed": args.seed, "total": len(results),
              "summary": summary, "cases": results}
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n[Summary] {len(results)} cases: {summary}")
    print(f"[Report] {path}")


if __name__ == "__main__":
    main()
