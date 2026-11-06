
import itertools
import re
from collections import defaultdict

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Boolean Simplifier",
    page_icon="∑",
    layout="wide"
)

# ---------------- DESIGN ----------------

st.markdown("""
<style>
.block-container {
    max-width: 1100px;
    padding-top: 2rem;
}
.hero {
    background: #17243a;
    padding: 28px;
    border-radius: 16px;
    border: 1px solid #334155;
    margin-bottom: 25px;
}
.hero h1 {
    font-size: 38px;
    margin-bottom: 8px;
}
.hero p {
    color: #cbd5e1;
}
.result {
    background: #123426;
    border: 1px solid #26865c;
    padding: 20px;
    border-radius: 12px;
    margin: 15px 0;
}
.result h2 {
    color: #86efac;
    font-size: 25px;
    overflow-wrap: anywhere;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>∑ Boolean Simplifier</h1>
<p>Minimize Boolean expressions using the Quine–McCluskey Tabular Method.</p>
</div>
""", unsafe_allow_html=True)


# ---------------- EXPRESSION PARSER ----------------

TOKEN_RE = re.compile(
    r"\s*(XNOR|XOR|AND|OR|NOT|[A-Za-z][A-Za-z0-9_]*|[01]|[()+*'!~&|^])",
    re.I
)


def tokenize(expression):
    tokens = []
    pos = 0

    while pos < len(expression):
        if expression[pos:].isspace():
            break

        match = TOKEN_RE.match(expression, pos)

        if not match:
            raise ValueError(
                "Invalid expression near: " + expression[pos:]
            )

        tokens.append(match.group(1).upper())
        pos = match.end()

    # Insert implicit AND for expressions such as AB, A'B, A(B+C).
    expanded = []

    def ends_operand(token):
        return (
            token not in (
                "AND", "OR", "XOR", "XNOR", "NOT",
                "(", "&", "|", "^", "+", "*", "!", "~"
            )
            or token in (")", "'")
        )

    def starts_operand(token):
        return token not in (
            "AND", "OR", "XOR", "XNOR",
            ")", "'", "&", "|", "^", "+", "*"
        )

    for token in tokens:
        if expanded:
            previous = expanded[-1]

            if ends_operand(previous) and starts_operand(token):
                expanded.append("AND")

        expanded.append(token)

    return expanded


class Parser:

    def __init__(self, tokens, values):
        self.tokens = tokens
        self.values = values
        self.pos = 0

    def peek(self):
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def eat(self):
        token = self.peek()
        if token is not None:
            self.pos += 1
        return token

    def parse(self):
        if not self.tokens:
            raise ValueError("Enter a Boolean expression.")

        result = self.parse_or()

        if self.peek() is not None:
            raise ValueError("Unexpected token: " + str(self.peek()))

        return int(bool(result))

    def parse_or(self):
        value = self.parse_xor()

        while self.peek() in ("OR", "|", "+"):
            self.eat()
            right = self.parse_xor()
            value = int(bool(value) or bool(right))

        return value

    def parse_xor(self):
        value = self.parse_and()

        while self.peek() in ("XOR", "XNOR", "^"):
            op = self.eat()
            right = self.parse_and()

            result = int(bool(value) ^ bool(right))

            if op == "XNOR":
                result = 1 - result

            value = result

        return value

    def parse_and(self):
        value = self.parse_unary()

        while self.peek() in ("AND", "&", "*"):
            self.eat()
            right = self.parse_unary()
            value = int(bool(value) and bool(right))

        return value

    def parse_unary(self):
        token = self.peek()

        if token in ("NOT", "!", "~"):
            self.eat()
            return 1 - self.parse_unary()

        if token == "(":
            self.eat()
            value = self.parse_or()

            if self.eat() != ")":
                raise ValueError("Missing closing parenthesis.")

        elif token in ("0", "1"):
            self.eat()
            value = int(token)

        elif token is None or token in (
            ")", "'", "AND", "OR", "XOR", "XNOR",
            "&", "|", "^", "+", "*"
        ):
            raise ValueError("Missing or unexpected operand.")

        else:
            self.eat()

            if token not in self.values:
                raise ValueError(
                    "Unknown variable: " + token
                )

            value = int(bool(self.values[token]))

        while self.peek() == "'":
            self.eat()
            value = 1 - value

        return value


def evaluate(expression, variables, bits):
    values = dict(zip(variables, bits))
    tokens = tokenize(expression)

    return Parser(tokens, values).parse()


# ---------------- TABULAR METHOD ----------------

def binary(number, width):
    return format(number, "0{}b".format(width))


def combine(a, b):
    difference = 0
    result = []

    for x, y in zip(a, b):

        if x == y:
            result.append(x)

        elif x != "-" and y != "-":
            difference += 1
            result.append("-")

        else:
            return None

    if difference == 1:
        return "".join(result)

    return None


def covers(pattern, number, width):
    bits = binary(number, width)

    return all(
        p == "-" or p == b
        for p, b in zip(pattern, bits)
    )


def minimum_cover(primes, chart, minterms):
    """Exact minimum set cover."""

    best = None
    best_score = (999, 999)

    def search(uncovered, selected):

        nonlocal best, best_score

        if not uncovered:
            score = (
                len(selected),
                sum(p.count("0") + p.count("1") for p in selected)
            )

            if score < best_score:
                best_score = score
                best = selected[:]

            return

        if len(selected) >= best_score[0]:
            return

        # Choose the uncovered minterm with the fewest choices.
        target = min(
            uncovered,
            key=lambda m: sum(
                m in chart[p] for p in primes if p not in selected
            )
        )

        options = [
            p for p in primes
            if p not in selected and target in chart[p]
        ]

        options.sort(
            key=lambda p: -len(chart[p] & uncovered)
        )

        for p in options:
            search(
                uncovered - chart[p],
                selected + [p]
            )

    search(set(minterms), [])

    return best or []


def tabular_method(minterms, dont_cares, width):

    if not minterms:
        return [], {}, []

    if len(minterms) == 2 ** width:
        return [], {}, ["-" * width]

    current = {
        binary(n, width)
        for n in set(minterms) | set(dont_cares)
    }

    rounds = []
    primes = set()
    round_number = 1

    while current:

        groups = defaultdict(list)

        for pattern in sorted(current):
            groups[pattern.count("1")].append(pattern)

        used = set()
        next_patterns = set()
        combinations = []

        for group in sorted(groups):

            for a in groups[group]:

                for b in groups.get(group + 1, []):

                    combined = combine(a, b)

                    if combined is not None:

                        used.add(a)
                        used.add(b)

                        next_patterns.add(combined)

                        combinations.append(
                            (a, b, combined)
                        )

        uncombined = sorted(current - used)
        primes.update(uncombined)

        rounds.append({
            "number": round_number,
            "groups": dict(groups),
            "combinations": combinations,
            "uncombined": uncombined
        })

        current = next_patterns
        round_number += 1

    chart = {}

    for p in sorted(primes):

        covered = {
            m for m in minterms
            if covers(p, m, width)
        }

        if covered:
            chart[p] = covered

    selected = minimum_cover(
        list(chart.keys()),
        chart,
        minterms
    )

    return rounds, chart, selected


# ---------------- EXPRESSION OUTPUT ----------------

def product_term(pattern, variables):

    if all(x == "-" for x in pattern):
        return "1"

    result = []

    for bit, variable in zip(pattern, variables):

        if bit == "1":
            result.append(variable)

        elif bit == "0":
            result.append(variable + "'")

    return "".join(result)


def final_expression(patterns, variables):

    if not patterns:
        return "0"

    if any(
        all(x == "-" for x in p)
        for p in patterns
    ):
        return "1"

    return " + ".join(
        product_term(p, variables)
        for p in patterns
    )


# ---------------- USER INPUT ----------------

left, right = st.columns([1.2, 0.8])

with left:

    st.subheader("1. Define your function")

    variable_input = st.text_input(
        "Variables (comma-separated)",
        "A, B, C"
    )

    expression = st.text_input(
        "Boolean Expression",
        "A AND B OR (NOT A AND C) XOR B"
    )

    st.caption(
        "Supports AND, OR, NOT, XOR, XNOR, "
        "parentheses, AB, A'B and (A+B)C."
    )

with right:

    st.subheader("2. Optional inputs")

    dc_input = st.text_input(
        "Don't-care minterms",
        "",
        placeholder="Example: 1, 3, 7"
    )

    st.caption(
        "Enter decimal minterm numbers separated by commas."
    )

    st.info(
        "The leftmost variable is the most significant bit."
    )


# ---------------- SIMPLIFY ----------------

if st.button(
    "Simplify Expression",
    type="primary",
    use_container_width=True
):

    try:

        variables = [
            x.strip().upper()
            for x in variable_input.split(",")
            if x.strip()
        ]

        if not variables:
            raise ValueError("Enter variables.")

        if len(variables) > 8:
            raise ValueError("Maximum 8 variables supported.")

        if len(set(variables)) != len(variables):
            raise ValueError("Variable names must be unique.")

        if any(
            not re.fullmatch(r"[A-Z][A-Z0-9_]*", x)
            for x in variables
        ):
            raise ValueError("Invalid variable name.")

        rows = []
        minterms = []

        for bits in itertools.product(
            (0, 1), repeat=len(variables)
        ):

            output = evaluate(
                expression,
                variables,
                bits
            )

            number = sum(
                bit << (len(variables) - 1 - i)
                for i, bit in enumerate(bits)
            )

            row = {"Minterm": number}

            row.update(dict(zip(variables, bits)))
            row["F"] = output

            rows.append(row)

            if output == 1:
                minterms.append(number)

        maximum = 2 ** len(variables) - 1

        if dc_input.strip():
            dont_cares = sorted(set(
                int(x.strip())
                for x in dc_input.split(",")
                if x.strip()
            ))
        else:
            dont_cares = []

        if any(
            n < 0 or n > maximum
            for n in dont_cares
        ):
            raise ValueError(
                "Don't-care values must be between 0 and {}.".format(
                    maximum
                )
            )

        if set(dont_cares) & set(minterms):
            raise ValueError(
                "A minterm cannot also be a don't-care."
            )

        rounds, chart, selected = tabular_method(
            minterms,
            dont_cares,
            len(variables)
        )

        answer = final_expression(
            selected,
            variables
        )

        st.session_state["result"] = {
            "variables": variables,
            "rows": rows,
            "minterms": minterms,
            "dont_cares": dont_cares,
            "rounds": rounds,
            "chart": chart,
            "selected": selected,
            "answer": answer
        }

    except Exception as error:
        st.error(str(error))


# ---------------- DISPLAY RESULTS ----------------

if "result" in st.session_state:

    r = st.session_state["result"]

    st.divider()

    st.subheader("Simplification Result")

    a, b, c = st.columns(3)

    a.metric("Variables", len(r["variables"]))
    b.metric("Minterms", len(r["minterms"]))
    c.metric("Prime Implicants", len(r["chart"]))

    st.markdown(
        '<div class="result">'
        '<p>FINAL MINIMIZED SOP EXPRESSION</p>'
        '<h2>{}</h2>'
        '</div>'.format(r["answer"]),
        unsafe_allow_html=True
    )

    st.code(r["answer"])

    tab1, tab2, tab3, tab4 = st.tabs([
        "Truth Table",
        "Tabulation Steps",
        "Prime Implicant Chart",
        "Minterms"
    ])

    # TRUTH TABLE

    with tab1:

        st.subheader("Complete Truth Table")

        df = pd.DataFrame(r["rows"])

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download CSV",
            df.to_csv(index=False),
            "truth_table.csv",
            "text/csv"
        )

    # TABULATION

    with tab2:

        st.subheader("Quine–McCluskey Reduction")

        for step in r["rounds"]:

            with st.expander(
                "Combination Round {}".format(step["number"]),
                expanded=True
            ):

                for group, patterns in sorted(
                    step["groups"].items()
                ):

                    st.write(
                        "**Group {}:** {}".format(
                            group,
                            ", ".join(patterns)
                        )
                    )

                if step["combinations"]:

                    st.write("**Combinations**")

                    st.dataframe(
                        pd.DataFrame(
                            step["combinations"],
                            columns=[
                                "Term 1",
                                "Term 2",
                                "Combined Term"
                            ]
                        ),
                        hide_index=True,
                        use_container_width=True
                    )

                else:
                    st.write("No further combinations.")

                st.write(
                    "**Uncombined terms:** " +
                    (", ".join(step["uncombined"]) or "None")
                )

    # PRIME IMPLICANT CHART

    with tab3:

        st.subheader("Prime Implicant Chart")

        if r["chart"]:

            data = []

            for pattern, covered in r["chart"].items():

                row = {
                    "Implicant": pattern,
                    "Product Term": product_term(
                        pattern,
                        r["variables"]
                    )
                }

                for m in r["minterms"]:
                    row["m{}".format(m)] = (
                        "✓" if m in covered else ""
                    )

                row["Selected"] = (
                    "✓" if pattern in r["selected"] else ""
                )

                data.append(row)

            st.dataframe(
                pd.DataFrame(data),
                hide_index=True,
                use_container_width=True
            )

        st.write("**Selected implicants:**")

        st.write(
            ", ".join(
                product_term(p, r["variables"])
                for p in r["selected"]
            ) or "None"
        )

    # MINTERMS

    with tab4:

        st.subheader("Function Representation")

        st.write("**Minterms:**", r["minterms"])

        st.write("**Don't-cares:**", r["dont_cares"])

        if r["minterms"]:

            st.latex(
                "F = \\\\Sigma m(" +
                ",".join(map(str, r["minterms"])) +
                ")"
            )

        else:
            st.latex("F = 0")


st.divider()

st.caption(
    "Boolean Simplifier | Tabular Minimization | "
    "Digital Electronics Learning Tool"
)
