import numpy as np
import sympy as sym


class MatrixUtils:
    @staticmethod
    def parse_matrix(text):
        text = text.strip()

        if not (text.startswith("[") and text.endswith("]")):
            raise ValueError("матрица должна быть записана в квадратных скобках")

        inner = text[1:-1].strip()
        if not inner:
            raise ValueError("матрица не должна быть пустой")

        rows_raw = [row.strip() for row in inner.split(";")]
        matrix = []

        for row in rows_raw:
            if not row:
                raise ValueError("обнаружена пустая строка")
            values = [x.strip() for x in row.split(",")]
            try:
                matrix.append([float(x.replace(" ", "")) for x in values])
            except ValueError:
                raise ValueError("все элементы матрицы должны быть числами")

        row_len = len(matrix[0])
        for row in matrix:
            if len(row) != row_len:
                raise ValueError("во всех строках должно быть одинаковое число элементов")

        return np.array(matrix, dtype=float)

    @staticmethod
    def format_number(x):
        if abs(x - int(x)) < 1e-10:
            return str(int(x))
        return f"{x:.6f}".rstrip("0").rstrip(".")

    @staticmethod
    def format_matrix(np_matrix):
        if np_matrix.size == 0:
            return ""

        str_matrix = [[MatrixUtils.format_number(x) for x in row] for row in np_matrix]
        widths = [max(len(str_matrix[i][j]) for i in range(len(str_matrix))) for j in range(len(str_matrix[0]))]
        lines = []
        for row in str_matrix:
            lines.append("  ".join(val.rjust(widths[j]) for j, val in enumerate(row)))
        return "\n".join(lines)

    @staticmethod
    def to_sympy_matrix(np_matrix):
        rows, cols = np_matrix.shape
        data = []
        for i in range(rows):
            row = []
            for j in range(cols):
                x = np_matrix[i, j]
                if abs(x - int(x)) < 1e-10:
                    row.append(sym.Integer(int(x)))
                else:
                    row.append(sym.nsimplify(x))
            data.append(row)
        return sym.Matrix(data)

    @staticmethod
    def format_sympy_number(x):
        x = sym.simplify(x)
        if x.is_real:
            try:
                xf = float(sym.N(x))
                if abs(xf - round(xf)) < 1e-10:
                    return str(int(round(xf)))
                return f"{xf:.6f}".rstrip("0").rstrip(".")
            except Exception:
                return str(x)
        return str(x)

    @staticmethod
    def format_sympy_matrix(M):
        rows = M.tolist()
        if not rows:
            return ""

        srows = [[MatrixUtils.format_sympy_number(x) for x in row] for row in rows]
        widths = [max(len(srows[i][j]) for i in range(len(srows))) for j in range(len(srows[0]))]

        lines = []
        for row in srows:
            lines.append("  ".join(val.rjust(widths[j]) for j, val in enumerate(row)))
        return "\n".join(lines)

    @staticmethod
    def format_complex(z):
        z = complex(z)
        if abs(z.imag) < 1e-10:
            return MatrixUtils.format_number(z.real)
        re = MatrixUtils.format_number(z.real)
        im = MatrixUtils.format_number(abs(z.imag))
        sign = "+" if z.imag >= 0 else "-"
        return f"{re} {sign} {im}j"

    @staticmethod
    def group_blocks_by_eigenvalue(blocks):
        groups = {}
        for block in blocks:
            lam = sym.simplify(block["lambda"])
            key = sym.sstr(lam)
            if key not in groups:
                groups[key] = {"lambda": lam, "blocks": []}
            groups[key]["blocks"].append(block)
        return groups

    @staticmethod
    def eigenvector_parametric_formula(rref_M, vec_index=1):
        """
        Строит параметрическую запись решения (A - λI)v = 0
        в виде:
        v_{11} = -v_{12} - v_{13}
        v_{12} = v_{12}
        v_{13} = v_{13}
        """
        n = rref_M.cols
        pivots = []
        pivot_set = set()

        for i in range(rref_M.rows):
            pivot_col = None
            nonzero_positions = [j for j in range(n) if sym.simplify(rref_M[i, j]) != 0]
            if nonzero_positions:
                pivot_col = nonzero_positions[0]
                pivots.append(pivot_col)
                pivot_set.add(pivot_col)

        free_cols = [j for j in range(n) if j not in pivot_set]

        left_names = [f"v_{{{vec_index}{j+1}}}" for j in range(n)]
        right_exprs = [None] * n

        for j in free_cols:
            right_exprs[j] = f"v_{{{vec_index}{j+1}}}"

        for row_i, pivot_j in enumerate(pivots):
            terms = []
            for j in free_cols:
                coef = sym.simplify(rref_M[row_i, j])
                if coef == 0:
                    continue

                coeff = sym.simplify(-coef)
                var = f"v_{{{vec_index}{j+1}}}"

                if coeff == 1:
                    terms.append(f"{var}")
                elif coeff == -1:
                    terms.append(f"-{var}")
                else:
                    terms.append(f"{MatrixUtils.format_sympy_number(coeff)}{var}")

            if not terms:
                right_exprs[pivot_j] = "0"
            else:
                expr = " + ".join(terms)
                expr = expr.replace("+ -", "- ")
                right_exprs[pivot_j] = expr

        return left_names, right_exprs

    @staticmethod
    def format_parametric_vector_formula(left_names, right_exprs):
        lines = []
        for l, r in zip(left_names, right_exprs):
            lines.append(f"{l} = {r}")
        return "\n".join(lines)

    @staticmethod
    def is_real_eigenvalue(lam):
        return abs(complex(sym.N(lam)).imag) < 1e-10

    @staticmethod
    def is_positive_imag_eigenvalue(lam):
        return complex(sym.N(lam)).imag > 1e-10

    @staticmethod
    def real_part_vector(v):
        return sym.Matrix([sym.simplify(sym.re(x)) for x in v])

    @staticmethod
    def imag_part_vector(v):
        return sym.Matrix([sym.simplify(sym.im(x)) for x in v])

    @staticmethod
    def real_jordan_block(lam, size):
        """
        Для вещественного λ: обычный жорданов блок size x size.
        Для комплексного λ = a + bi (b>0): вещественный блок размера 2*size x 2*size:
        [A2 I2 0 ...]
        [0  A2 I2...]
        ...
        где A2 = [[a,b],[-b,a]]
        """
        lam_c = complex(sym.N(lam))
        a = sym.nsimplify(lam_c.real)
        b = sym.nsimplify(abs(lam_c.imag))

        if abs(lam_c.imag) < 1e-10:
            J = sym.zeros(size, size)
            for i in range(size):
                J[i, i] = a
                if i < size - 1:
                    J[i, i + 1] = 1
            return J

        J = sym.zeros(2 * size, 2 * size)
        A2 = sym.Matrix([[a, b], [-b, a]])
        I2 = sym.eye(2)

        for k in range(size):
            r = 2 * k
            J[r:r+2, r:r+2] = A2
            if k < size - 1:
                J[r:r+2, r+2:r+4] = I2

        return J