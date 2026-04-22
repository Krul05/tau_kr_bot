import itertools
import numpy as np
import sympy as sp
from scipy.linalg import solve_sylvester

from first_solver import FirstSolver
from matrix_utils import MatrixUtils


class SecondSolver:
    TOL = 1e-8

    @staticmethod
    def is_stabilizable(A, B):
        return FirstSolver.jordan_analysis(A, B)["stabilizable"]

    @staticmethod
    def controllability_matrix(A, B):
        n = A.shape[0]
        U = B.copy()
        cur = B.copy()
        for _ in range(1, n):
            cur = A @ cur
            U = np.hstack((U, cur))
        return U

    @staticmethod
    def is_controllable(A, B, tol=1e-8):
        U = SecondSolver.controllability_matrix(A, B)
        return np.linalg.matrix_rank(U, tol) == A.shape[0]

    @staticmethod
    def observability_matrix(Y, Gamma):
        n = Gamma.shape[0]
        rows = [Y]
        cur = Y.copy()
        for _ in range(1, n):
            cur = cur @ Gamma
            rows.append(cur)
        return np.vstack(rows)

    @staticmethod
    def is_observable_pair(Y, Gamma, tol=1e-8):
        O = SecondSolver.observability_matrix(Y, Gamma)
        return np.linalg.matrix_rank(O, tol) == Gamma.shape[0]

    @staticmethod
    def choose_observable_Y(Gamma, tol=1e-8):
        if Gamma is None:
            raise ValueError("матрица Γ не задана")

        Gamma = np.array(Gamma)
        if Gamma.ndim != 2 or Gamma.shape[0] != Gamma.shape[1]:
            raise ValueError("матрица Γ должна быть квадратной")

        n = Gamma.shape[0]

        # Сначала пробуем Y, состоящий только из единиц
        Y = np.ones((1, n), dtype=float)
        if SecondSolver.is_observable_pair(Y, Gamma, tol):
            return Y

        # Потом пробуем базисные строки
        for i in range(n):
            Y = np.zeros((1, n), dtype=float)
            Y[0, i] = 1.0
            if SecondSolver.is_observable_pair(Y, Gamma, tol):
                return Y

        # Потом простые комбинации
        for vals in itertools.product([-1, 0, 1], repeat=n):
            if all(v == 0 for v in vals):
                continue
            Y = np.array([vals], dtype=float)
            if SecondSolver.is_observable_pair(Y, Gamma, tol):
                return Y

        raise ValueError("не удалось подобрать Y так, чтобы пара (Y, Γ) была наблюдаемой")

    @staticmethod
    def parse_poles(poles_input):
        if poles_input is None:
            raise ValueError("спектр не введён")

        text = poles_input.strip()
        if not text:
            raise ValueError("спектр пустой")

        parts = [p.strip().replace("i", "j") for p in text.split(",")]

        if any(p == "" for p in parts):
            raise ValueError(
                "неверный формат спектра: используйте запись вида -1, -2, -2 или -1+1j, -1-1j"
            )

        poles = []
        for p in parts:
            try:
                val = complex(sp.sympify(p))
            except Exception:
                raise ValueError(
                    f"не удалось распознать собственное число '{p}'. "
                    "Используйте запись вида -1, -2, -2 или -1+1j, -1-1j"
                )

            if abs(val.real) > 1e12 or abs(val.imag) > 1e12:
                raise ValueError(f"слишком большое по модулю значение '{p}'")

            poles.append(val)

        return poles

    @staticmethod
    def default_poles(n):
        if n == 1:
            return [-1]
        if n == 2:
            return [-2, -2]
        if n == 3:
            return [-1, -2, -2]

        poles = [-2, -2]
        cur = -3
        while len(poles) < n:
            poles.append(cur)
            cur -= 1
        return poles[:n]

    @staticmethod
    def build_gamma_from_poles(poles, tol=1e-8):
        """
        Строит Γ в правильной вещественной форме.

        1) Для кратных вещественных полюсов строит один жорданов блок:
           [λ 1 0 ...]
           [0 λ 1 ...]
           [...]
        2) Для комплексно-сопряжённой пары a±jb строит вещественный блок:
           [[a, b],
            [-b, a]]

        Это ближе к презентации: для кратных полюсов Γ должна быть не просто
        диагональной, а жордановой. Тогда условие наблюдаемости пары (Y, Γ)
        работает корректнее.
        """
        poles = [complex(p) for p in poles]
        used = [False] * len(poles)

        real_groups = {}
        complex_blocks = []

        for i, p in enumerate(poles):
            if used[i]:
                continue

            # Вещественный полюс
            if abs(p.imag) < tol:
                key = round(p.real, 12)
                if key not in real_groups:
                    real_groups[key] = {"lam": p.real, "mult": 0}
                real_groups[key]["mult"] += 1
                used[i] = True
                continue

            # Комплексный полюс: ищем сопряжённый
            pair_idx = None
            for j in range(i + 1, len(poles)):
                if used[j]:
                    continue
                q = poles[j]
                if abs(q.real - p.real) < tol and abs(q.imag + p.imag) < tol:
                    pair_idx = j
                    break

            if pair_idx is None:
                raise ValueError(
                    f"для комплексного полюса {p} не найдена сопряжённая пара. "
                    "Для вещественной Γ комплексные полюса нужно задавать сопряжёнными парами."
                )

            a = p.real
            b = abs(p.imag)
            complex_blocks.append(np.array([[a, b], [-b, a]], dtype=float))
            used[i] = True
            used[pair_idx] = True

        blocks = []

        # Жордановы блоки для вещественных полюсов с учётом кратности
        for key in sorted(real_groups.keys()):
            lam = real_groups[key]["lam"]
            mult = real_groups[key]["mult"]

            J = np.zeros((mult, mult), dtype=float)
            for i in range(mult):
                J[i, i] = lam
                if i < mult - 1:
                    J[i, i + 1] = 1.0

            blocks.append(J)

        # Вещественные блоки для комплексно-сопряжённых пар
        blocks.extend(complex_blocks)

        n = sum(block.shape[0] for block in blocks)
        Gamma = np.zeros((n, n), dtype=float)

        idx = 0
        for block in blocks:
            m = block.shape[0]
            Gamma[idx:idx + m, idx:idx + m] = block
            idx += m

        return Gamma

    @staticmethod
    def solve_sylvester_numeric(A, B, Gamma, Y):
        RHS = B @ Y
        P = solve_sylvester(A, -Gamma, RHS)
        return np.array(P, dtype=float if np.isrealobj(P) else complex)

    @staticmethod
    def jordan_decomposition_numeric(A, tol=1e-8):
        vals, vecs = np.linalg.eig(A)
        n = A.shape[0]

        if np.linalg.matrix_rank(vecs, tol) < n:
            raise ValueError(
                "не удалось построить невырожденную матрицу собственных векторов для перехода к жордановой форме"
            )

        P = vecs
        P_inv = np.linalg.inv(P)
        J = P_inv @ A @ P
        J[np.abs(J) < tol] = 0
        return J, P

    @staticmethod
    def find_uncontrollable_indices_in_jordan(J, B_j, tol=1e-8):
        """
        Определяет индексы неуправляемых мод в жордановой форме.

        Исправление:
        - для вещественной 2x2 клетки, соответствующей комплексно-сопряжённой паре,
          считаем пару неуправляемой только если ОБЕ строки блока в B_j нулевые;
        - для комплексной диагональной формы с парой λ, λ̄ делаем то же самое;
        - для обычной вещественной моды оставляем старую логику.
        """
        n = J.shape[0]
        indices = []
        i = 0

        while i < n:
            # --- Случай 1: вещественный 2x2 блок для комплексно-сопряжённой пары ---
            if i + 1 < n:
                block = J[i:i + 2, i:i + 2]

                is_real_complex_block = (
                        abs(np.imag(block)).max() < tol
                        and abs(block[0, 0] - block[1, 1]) < tol
                        and abs(block[0, 1]) > tol
                        and abs(block[1, 0]) > tol
                        and abs(block[0, 1] + block[1, 0]) < tol
                )

                if is_real_complex_block:
                    if np.linalg.norm(B_j[i:i + 2, :]) < tol:
                        indices.extend([i, i + 1])
                    i += 2
                    continue

            # --- Случай 2: комплексная диагональная форма λ, λ̄ ---
            if i + 1 < n:
                lam1 = J[i, i]
                lam2 = J[i + 1, i + 1]

                is_complex_conjugate_pair = (
                        abs(np.imag(lam1)) > tol
                        and abs(lam2 - np.conjugate(lam1)) < tol
                        and abs(J[i, i + 1]) < tol
                        and abs(J[i + 1, i]) < tol
                )

                if is_complex_conjugate_pair:
                    if np.linalg.norm(B_j[i:i + 2, :]) < tol:
                        indices.extend([i, i + 1])
                    i += 2
                    continue

            # --- Случай 3: обычная вещественная мода ---
            if np.linalg.norm(B_j[i, :]) < tol:
                indices.append(i)

            i += 1

        return indices

    @staticmethod
    def keep_controllable_part(J, B_j, uncontrollable_indices):
        keep = [i for i in range(J.shape[0]) if i not in uncontrollable_indices]
        J_star = J[np.ix_(keep, keep)]
        B_star = B_j[keep, :]
        return J_star, B_star, keep

    @staticmethod
    def expand_gain(K_star, keep, n):
        K_full = np.zeros((1, n), dtype=complex)
        for local_idx, global_idx in enumerate(keep):
            K_full[0, global_idx] = K_star[0, local_idx]
        return K_full

    @staticmethod
    def _cleanup_complex(arr, tol=1e-9):
        arr = np.array(arr, dtype=complex)
        arr.real[np.abs(arr.real) < tol] = 0
        arr.imag[np.abs(arr.imag) < tol] = 0
        return np.real_if_close(arr, tol=1000)

    @staticmethod
    def _spectra_do_not_intersect(spec1, spec2, tol=1e-7):
        for a in spec1:
            for b in spec2:
                if abs(a - b) < tol:
                    return False
        return True

    @staticmethod
    def _sort_complex_list(vals, tol=1e-7):
        cleaned = []
        for z in vals:
            z = complex(z)
            re = 0.0 if abs(z.real) < tol else z.real
            im = 0.0 if abs(z.imag) < tol else z.imag
            cleaned.append(complex(re, im))
        return sorted(cleaned, key=lambda z: (round(z.real, 7), round(z.imag, 7)))

    @staticmethod
    def _spectra_match(spec1, spec2, tol=1e-6):
        a = SecondSolver._sort_complex_list(spec1, tol)
        b = SecondSolver._sort_complex_list(spec2, tol)
        if len(a) != len(b):
            return False
        return all(abs(x - y) < tol for x, y in zip(a, b))

    @staticmethod
    def _format_spectrum(vals):
        return ", ".join(MatrixUtils.format_complex(v) for v in vals)

    @staticmethod
    def solve_order_1(A, B, desired_poles):
        lam_des = complex(desired_poles[0])
        a11 = complex(A[0, 0])
        b11 = complex(B[0, 0])

        if abs(b11) < 1e-12:
            raise ValueError(
                "для скалярной системы коэффициент B равен нулю, поэтому невозможно изменить собственное число"
            )

        k1 = (lam_des - a11) / b11
        K = np.array([[k1]], dtype=complex)
        Acl = A + B @ K
        eig_cl = np.linalg.eigvals(Acl)

        if not SecondSolver._spectra_match(eig_cl, [lam_des]):
            raise ValueError(
                "проверка не пройдена: после построения K спектр матрицы A + BK "
                "не совпал с желаемым спектром"
            )

        result = []
        result.append("Порядок системы равен 1, поэтому решаем задачу напрямую.")
        result.append("")
        result.append("Ищем управление в виде")
        result.append("<pre>u = Kx = k1 x</pre>")
        result.append("Тогда")
        result.append("<pre>Aз = A + BK</pre>")
        result.append("")
        result.append("Матрица K:")
        result.append(f"<pre>{MatrixUtils.format_matrix(SecondSolver._cleanup_complex(K))}</pre>")
        result.append("")
        result.append("Матрица A + BK:")
        result.append(f"<pre>{MatrixUtils.format_matrix(SecondSolver._cleanup_complex(Acl))}</pre>")
        result.append("")
        result.append("Проверка:")
        result.append("полученный спектр совпал с желаемым.")
        for lam in eig_cl:
            result.append(f"λ = {MatrixUtils.format_complex(lam)}")

        return K, Acl, "\n".join(result)

    @staticmethod
    def solve_order_2_trace_det(A, B, desired_poles):
        if len(desired_poles) != 2:
            raise ValueError("для системы второго порядка нужно задать ровно 2 собственных числа")

        a11, a12 = complex(A[0, 0]), complex(A[0, 1])
        a21, a22 = complex(A[1, 0]), complex(A[1, 1])
        b1, b2 = complex(B[0, 0]), complex(B[1, 0])

        k1, k2 = sp.symbols("k1 k2")
        Acl_sym = sp.Matrix([
            [sp.nsimplify(a11) + sp.nsimplify(b1) * k1, sp.nsimplify(a12) + sp.nsimplify(b1) * k2],
            [sp.nsimplify(a21) + sp.nsimplify(b2) * k1, sp.nsimplify(a22) + sp.nsimplify(b2) * k2],
        ])

        lam1 = sp.nsimplify(desired_poles[0])
        lam2 = sp.nsimplify(desired_poles[1])

        tr_eq = sp.Eq(Acl_sym.trace(), sp.expand(lam1 + lam2))
        det_eq = sp.Eq(sp.expand(Acl_sym.det()), sp.expand(lam1 * lam2))

        sol = sp.solve((tr_eq, det_eq), (k1, k2), dict=True)
        if not sol:
            raise ValueError(
                "не удалось решить систему по следу и определителю. "
                "Значит, для заданного спектра и этой пары (A, B) подходящий регулятор в таком виде не найден"
            )

        sol = sol[0]
        K = np.array([[complex(sp.N(sol[k1])), complex(sp.N(sol[k2]))]], dtype=complex)
        Acl = A + B @ K
        eig_cl = np.linalg.eigvals(Acl)

        if not SecondSolver._spectra_match(eig_cl, desired_poles):
            raise ValueError(
                "проверка не пройдена: после решения по следу и определителю "
                "спектр матрицы A + BK не совпал с желаемым"
            )

        s = complex(desired_poles[0] + desired_poles[1])
        p = complex(desired_poles[0] * desired_poles[1])
        disc = s * s - 4 * p

        result = []
        result.append("Порядок системы равен 2, поэтому решаем задачу через след и определитель.")
        result.append("")
        result.append("Ищем управление в виде")
        result.append("<pre>u = Kx = k1 x1 + k2 x2</pre>")
        result.append("")
        result.append("Тогда")
        result.append("<pre>Aз = A + BK</pre>")
        result.append("")
        result.append("Используем свойства:")
        result.append("<pre>tr(A + BK) = λ1 + λ2</pre>")
        result.append("<pre>det(A + BK) = λ1 λ2</pre>")
        result.append("")
        result.append("Дискриминант желаемого характеристического полинома:")
        result.append(f"<pre>D = (λ1 + λ2)^2 - 4 λ1 λ2 = {MatrixUtils.format_complex(disc)}</pre>")
        result.append("")
        result.append("Составляем систему уравнений:")
        result.append(f"<pre>{sp.sstr(sp.expand(tr_eq.lhs))} = {sp.sstr(sp.expand(tr_eq.rhs))}</pre>")
        result.append(f"<pre>{sp.sstr(sp.expand(det_eq.lhs))} = {sp.sstr(sp.expand(det_eq.rhs))}</pre>")
        result.append("")
        result.append("Матрица K:")
        result.append(f"<pre>{MatrixUtils.format_matrix(SecondSolver._cleanup_complex(K))}</pre>")
        result.append("")
        result.append("Матрица A + BK:")
        result.append(f"<pre>{MatrixUtils.format_matrix(SecondSolver._cleanup_complex(Acl))}</pre>")
        result.append("")
        result.append("Проверка:")
        result.append("полученный спектр совпал с желаемым.")
        for lam in eig_cl:
            result.append(f"λ = {MatrixUtils.format_complex(lam)}")

        return K, Acl, "\n".join(result)

    def solve_full_small_order(self, A, B, desired_poles):
        n = A.shape[0]
        if n == 1:
            return self.solve_order_1(A, B, desired_poles)
        if n == 2:
            return self.solve_order_2_trace_det(A, B, desired_poles)
        raise ValueError("метод для малых порядков применим только при n <= 2")

    def solve_full_sylvester(self, A, B, desired_poles):
        n = A.shape[0]

        if not self.is_controllable(A, B):
            raise ValueError(
                "не выполнено условие метода Сильвестра из презентации: пара (A, B) не полностью управляема"
            )

        if n <= 2:
            return self.solve_full_small_order(A, B, desired_poles)

        Gamma = self.build_gamma_from_poles(desired_poles)

        eig_A = np.linalg.eigvals(A)
        eig_Gamma = np.linalg.eigvals(Gamma)

        if not self._spectra_do_not_intersect(eig_A, eig_Gamma):
            raise ValueError(
                "не выполнено условие σ(A) ∩ σ(Γ) = ∅. "
                f"Спектр A: [{self._format_spectrum(eig_A)}]. "
                f"Спектр Γ: [{self._format_spectrum(eig_Gamma)}]. "
                "Из-за пересечения спектров метод Сильвестра в этой постановке применять нельзя."
            )

        Y = self.choose_observable_Y(Gamma)

        if not self.is_observable_pair(Y, Gamma):
            raise ValueError(
                "не выполнено условие метода Сильвестра: пара (Y, Γ) не наблюдаема"
            )

        P = self.solve_sylvester_numeric(A, B, Gamma, Y)

        if not np.all(np.isfinite(P)):
            raise ValueError(
                "при решении уравнения Сильвестра получены некорректные значения матрицы P"
            )

        if np.linalg.matrix_rank(P, self.TOL) < n:
            raise ValueError(
                "матрица P оказалась вырожденной, поэтому невозможно вычислить K = -YP^-1"
            )

        K = -Y @ np.linalg.inv(P)
        Acl = A + B @ K
        eig_cl = np.linalg.eigvals(Acl)

        if not self._spectra_match(eig_cl, desired_poles):
            raise ValueError(
                "проверка не пройдена: полученный спектр матрицы A + BK не совпал с желаемым. "
                f"Желаемый спектр: [{self._format_spectrum(desired_poles)}]. "
                f"Полученный спектр: [{self._format_spectrum(eig_cl)}]. "
                "Если все теоретические условия соблюдены, а проверка не проходит, "
                "значит проблема обычно связана с численной неточностью или неудачной постановкой Γ"
            )

        result = []
        result.append("Система полностью управляема.")
        result.append("Так как порядок системы больше 2, синтезируем регулятор по матричному уравнению Сильвестра.")
        result.append("")
        result.append("Ищем управление в виде")
        result.append("<pre>u = Kx</pre>")
        result.append("Тогда матрица замкнутой системы")
        result.append("<pre>Aз = A + BK</pre>")
        result.append("")
        result.append("Проверяем условия метода Сильвестра:")
        result.append("<pre>σ(A) ∩ σ(Γ) = ∅</pre>")
        result.append("<pre>(A, B) — управляема</pre>")
        result.append("<pre>(Y, Γ) — наблюдаема</pre>")
        result.append("")
        result.append("Все проверки выполнены.")
        result.append("")
        result.append("Матрица Γ:")
        result.append(f"<pre>{MatrixUtils.format_matrix(self._cleanup_complex(Gamma))}</pre>")
        result.append("")
        result.append("Матрица Y:")
        result.append(f"<pre>{MatrixUtils.format_matrix(self._cleanup_complex(Y))}</pre>")
        result.append("")
        result.append("Решаем уравнение Сильвестра")
        result.append("<pre>AP - PΓ = BY</pre>")
        result.append("и затем находим")
        result.append("<pre>K = -YP^-1</pre>")
        result.append("")
        result.append("Матрица P:")
        result.append(f"<pre>{MatrixUtils.format_matrix(self._cleanup_complex(P))}</pre>")
        result.append("")
        result.append("Матрица K:")
        result.append(f"<pre>{MatrixUtils.format_matrix(self._cleanup_complex(K))}</pre>")
        result.append("")
        result.append("Матрица A + BK:")
        result.append(f"<pre>{MatrixUtils.format_matrix(self._cleanup_complex(Acl))}</pre>")
        result.append("")
        result.append("Проверка:")
        result.append("полученный спектр совпал с желаемым.")
        for lam in eig_cl:
            result.append(f"λ = {MatrixUtils.format_complex(lam)}")

        return K, Acl, "\n".join(result)

    def solve_with_truncation(self, A, B, desired_poles):
        n = A.shape[0]

        J, P = self.jordan_decomposition_numeric(A)
        P_inv = np.linalg.inv(P)
        B_j = P_inv @ B

        uncontrollable_indices = self.find_uncontrollable_indices_in_jordan(J, B_j)

        if len(uncontrollable_indices) == 0:
            return self.solve_full_sylvester(A, B, desired_poles)

        for idx in uncontrollable_indices:
            lam = J[idx, idx]
            if np.real(lam) >= 0:
                raise ValueError(
                    "система не стабилизируема: среди неуправляемых мод есть неустойчивые собственные числа"
                )

        J_star, B_star, keep = self.keep_controllable_part(J, B_j, uncontrollable_indices)
        r = J_star.shape[0]

        if len(desired_poles) != r:
            raise ValueError(
                f"после усечения порядок управляемой подсистемы равен {r}, "
                f"поэтому нужно задать ровно {r} собственных чисел"
            )

        if not self.is_controllable(J_star, B_star):
            raise ValueError(
                "после усечения подсистема должна быть полностью управляема, но проверка управляемости не пройдена"
            )

        result = []
        result.append("Система не полностью управляема, но стабилизируема.")
        result.append("Поэтому выполняем синтез через усечение.")
        result.append("")
        result.append("1) Переходим к жордановой форме в координатах")
        result.append("<pre>x^ = P^-1 x</pre>")
        result.append("<pre>x = P x^</pre>")
        result.append("Тогда")
        result.append("<pre>x^dot = AJ x^ + BJ u</pre>")
        result.append("<pre>AJ = P^-1 A P</pre>")
        result.append("<pre>BJ = P^-1 B</pre>")
        result.append("")
        result.append("2) Вычёркиваем неуправляемые устойчивые моды.")
        result.append("3) Получаем управляемую подсистему меньшего порядка.")
        result.append("")
        result.append("Матрица AJ:")
        result.append(f"<pre>{MatrixUtils.format_matrix(self._cleanup_complex(J))}</pre>")
        result.append("")
        result.append("Матрица BJ:")
        result.append(f"<pre>{MatrixUtils.format_matrix(self._cleanup_complex(B_j))}</pre>")
        result.append("")
        result.append("Матрица AJ*:")
        result.append(f"<pre>{MatrixUtils.format_matrix(self._cleanup_complex(J_star))}</pre>")
        result.append("")
        result.append("Матрица BJ*:")
        result.append(f"<pre>{MatrixUtils.format_matrix(self._cleanup_complex(B_star))}</pre>")
        result.append("")

        if r <= 2:
            K_j_star, _, small_text = self.solve_full_small_order(J_star, B_star, desired_poles)
            result.append("Так как порядок усечённой подсистемы меньше или равен 2, решаем через след и определитель.")
            result.append("")
            result.append(small_text)
        else:
            Gamma_star = self.build_gamma_from_poles(desired_poles)

            eig_Js = np.linalg.eigvals(J_star)
            eig_Gs = np.linalg.eigvals(Gamma_star)

            if not self._spectra_do_not_intersect(eig_Js, eig_Gs):
                raise ValueError(
                    "для усечённой подсистемы не выполнено условие σ(AJ*) ∩ σ(Γ*) = ∅. "
                    f"Спектр AJ*: [{self._format_spectrum(eig_Js)}]. "
                    f"Спектр Γ*: [{self._format_spectrum(eig_Gs)}]. "
                    "Из-за пересечения спектров метод Сильвестра в этой постановке применять нельзя."
                )

            Y_star = self.choose_observable_Y(Gamma_star)

            if not self.is_observable_pair(Y_star, Gamma_star):
                raise ValueError(
                    "для усечённой подсистемы не выполнено условие наблюдаемости пары (Y*, Γ*)"
                )

            P_star = self.solve_sylvester_numeric(J_star, B_star, Gamma_star, Y_star)

            if not np.all(np.isfinite(P_star)):
                raise ValueError(
                    "для усечённой подсистемы при решении Сильвестра получены некорректные значения P*"
                )

            if np.linalg.matrix_rank(P_star, self.TOL) < r:
                raise ValueError(
                    "после усечения матрица P* вырождена, поэтому невозможно вычислить KJ* = -Y*(P*)^-1"
                )

            K_j_star = -Y_star @ np.linalg.inv(P_star)
            Acl_star = J_star + B_star @ K_j_star
            eig_star = np.linalg.eigvals(Acl_star)

            if not self._spectra_match(eig_star, desired_poles):
                raise ValueError(
                    "для усечённой подсистемы проверка не пройдена: "
                    "полученный спектр не совпал с желаемым"
                )

            result.append("Для усечённой подсистемы применяем Сильвестра.")
            result.append("Проверки выполнены:")
            result.append("<pre>σ(AJ*) ∩ σ(Γ*) = ∅</pre>")
            result.append("<pre>(AJ*, BJ*) — управляема</pre>")
            result.append("<pre>(Y*, Γ*) — наблюдаема</pre>")
            result.append("")
            result.append("Матрица Γ*:")
            result.append(f"<pre>{MatrixUtils.format_matrix(self._cleanup_complex(Gamma_star))}</pre>")
            result.append("")
            result.append("Матрица Y*:")
            result.append(f"<pre>{MatrixUtils.format_matrix(self._cleanup_complex(Y_star))}</pre>")
            result.append("")
            result.append("Матрица P*:")
            result.append(f"<pre>{MatrixUtils.format_matrix(self._cleanup_complex(P_star))}</pre>")
            result.append("")
            result.append("Регулятор для усечённой подсистемы KJ*:")
            result.append(f"<pre>{MatrixUtils.format_matrix(self._cleanup_complex(K_j_star))}</pre>")
            result.append("")

        K_j = self.expand_gain(K_j_star, keep, n)
        K = K_j @ P_inv
        Acl = A + B @ K
        eig_cl = np.linalg.eigvals(Acl)

        result.append("Дополняем регулятор нулями до KJ.")
        result.append("Так как")
        result.append("<pre>u = KJ x^ = KJ P^-1 x</pre>")
        result.append("то регулятор в исходных координатах равен")
        result.append("<pre>K = KJ P^-1</pre>")
        result.append("")
        result.append("Матрица K:")
        result.append(f"<pre>{MatrixUtils.format_matrix(self._cleanup_complex(K))}</pre>")
        result.append("")
        result.append("Матрица A + BK:")
        result.append(f"<pre>{MatrixUtils.format_matrix(self._cleanup_complex(Acl))}</pre>")
        result.append("")
        result.append("Собственные числа замкнутой системы:")
        for lam in eig_cl:
            result.append(f"λ = {MatrixUtils.format_complex(lam)}")

        return K, Acl, "\n".join(result)

    def reduced_order_for_second_task(self, A, B):
        if self.is_controllable(A, B):
            return A.shape[0]

        J, P = self.jordan_decomposition_numeric(A)
        P_inv = np.linalg.inv(P)
        B_j = P_inv @ B
        uncontrollable_indices = self.find_uncontrollable_indices_in_jordan(J, B_j)
        keep = [i for i in range(A.shape[0]) if i not in uncontrollable_indices]
        return len(keep)

    def solve(self, A, B, poles_input=None):
        n = A.shape[0]

        if A.shape[0] != A.shape[1]:
            return "Ошибка: матрица A должна быть квадратной."

        if B.shape[0] != n:
            return "Ошибка: число строк матрицы B должно совпадать с размерностью матрицы A."

        if B.shape[1] != 1:
            return "Ошибка: в задании 2 матрица B должна быть столбцом."

        try:
            stabilizable = self.is_stabilizable(A, B)
        except Exception as e:
            return f"Ошибка при анализе стабилизируемости: {e}"

        if not stabilizable:
            return (
                "Решение задания 2\n\n"
                "Синтез модального регулятора невозможен.\n"
                "Причина: система не стабилизируема."
            )

        try:
            desired_poles = (
                self.parse_poles(poles_input)
                if poles_input and str(poles_input).strip()
                else self.default_poles(n)
            )
        except Exception as e:
            return (
                "Решение задания 2\n\n"
                "Не удалось разобрать желаемый спектр.\n"
                f"Причина: {e}"
            )

        try:
            result = ["Решение задания 2", ""]

            if self.is_controllable(A, B):
                if len(desired_poles) != n:
                    return (
                        "Решение задания 2\n\n"
                        f"Ошибка: для системы порядка {n} нужно задать ровно {n} собственных чисел."
                    )
                _, _, text = self.solve_full_sylvester(A, B, desired_poles)
                result.append(text)
            else:
                _, _, text = self.solve_with_truncation(A, B, desired_poles)
                result.append(text)

            return "\n".join(result)

        except Exception as e:
            return (
                "Решение задания 2\n\n"
                "Не удалось построить модальный регулятор.\n"
                f"Причина: {e}"
            )