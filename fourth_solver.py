import itertools
import numpy as np
import sympy as sp
from scipy.linalg import solve_sylvester

from matrix_utils import MatrixUtils
from third_solver import ThirdSolver


class FourthSolver:
    TOL = 1e-8

    @staticmethod
    def is_detectable(A, C):
        return ThirdSolver.jordan_observability_analysis(A, C)["detectable"]

    @staticmethod
    def observability_matrix(A, C):
        n = A.shape[0]
        V = C.copy()
        cur = C.copy()
        for _ in range(1, n):
            cur = cur @ A
            V = np.vstack((V, cur))
        return V

    @staticmethod
    def is_observable(A, C, tol=1e-8):
        V = FourthSolver.observability_matrix(A, C)
        return np.linalg.matrix_rank(V, tol) == A.shape[0]

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
        U = FourthSolver.controllability_matrix(A, B)
        return np.linalg.matrix_rank(U, tol) == A.shape[0]

    @staticmethod
    def choose_controllable_Y(Gamma, tol=1e-8):
        if Gamma is None:
            raise ValueError("матрица Γ не задана")

        Gamma = np.array(Gamma)
        if Gamma.ndim != 2 or Gamma.shape[0] != Gamma.shape[1]:
            raise ValueError("матрица Γ должна быть квадратной")

        n = Gamma.shape[0]

        # Сначала пробуем Y, состоящий только из единиц
        Y = np.ones((n, 1), dtype=float)
        if FourthSolver.is_controllable(Gamma, Y, tol):
            return Y

        # Потом пробуем базисные столбцы
        for i in range(n):
            Y = np.zeros((n, 1), dtype=float)
            Y[i, 0] = 1.0
            if FourthSolver.is_controllable(Gamma, Y, tol):
                return Y

        # Потом простые комбинации
        for vals in itertools.product([-1, 0, 1], repeat=n):
            if all(v == 0 for v in vals):
                continue
            Y = np.array(vals, dtype=float).reshape(n, 1)
            if FourthSolver.is_controllable(Gamma, Y, tol):
                return Y

        raise ValueError("не удалось подобрать Y так, чтобы пара (Γ, Y) была управляема")

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
                "неверный формат спектра: используйте запись вида -1, -2-1j, -2+1j"
            )

        poles = []
        for p in parts:
            try:
                val = complex(sp.sympify(p))
            except Exception:
                raise ValueError(
                    f"не удалось распознать собственное число '{p}'. "
                    "Используйте запись вида -1, -2-1j, -2+1j"
                )

            if abs(val.real) > 1e12 or abs(val.imag) > 1e12:
                raise ValueError(f"слишком большое по модулю значение '{p}'")

            poles.append(val)

        return poles

    @staticmethod
    def default_observer_poles(n):
        if n == 1:
            return [-2]
        if n == 2:
            return [-2 - 1j, -2 + 1j]
        if n == 3:
            return [-1, -2 - 1j, -2 + 1j]

        poles = [-2 - 1j, -2 + 1j]
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
        """
        poles = [complex(p) for p in poles]
        used = [False] * len(poles)

        real_groups = {}
        complex_blocks = []

        for i, p in enumerate(poles):
            if used[i]:
                continue

            if abs(p.imag) < tol:
                key = round(p.real, 12)
                if key not in real_groups:
                    real_groups[key] = {"lam": p.real, "mult": 0}
                real_groups[key]["mult"] += 1
                used[i] = True
                continue

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

        for key in sorted(real_groups.keys()):
            lam = real_groups[key]["lam"]
            mult = real_groups[key]["mult"]

            J = np.zeros((mult, mult), dtype=float)
            for i in range(mult):
                J[i, i] = lam
                if i < mult - 1:
                    J[i, i + 1] = 1.0
            blocks.append(J)

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
    def solve_sylvester_numeric_observer(A, C, Gamma, Y):
        """
        Решает уравнение
            Q A - Γ Q = Y C
        как
            (-Γ)Q + Q A = Y C
        """
        RHS = Y @ C
        Q = solve_sylvester(-Gamma, A, RHS)
        return np.array(Q, dtype=float if np.isrealobj(Q) else complex)

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
    def find_unobservable_indices_in_jordan(J, C_j, tol=1e-8):
        """
        Для комплексно-сопряжённой пары считаем её ненаблюдаемой
        только если обе соответствующие колонки в C_j нулевые.
        """
        n = J.shape[0]
        indices = []
        i = 0

        while i < n:
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
                    if np.linalg.norm(C_j[:, i:i + 2]) < tol:
                        indices.extend([i, i + 1])
                    i += 2
                    continue

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
                    if np.linalg.norm(C_j[:, i:i + 2]) < tol:
                        indices.extend([i, i + 1])
                    i += 2
                    continue

            if np.linalg.norm(C_j[:, i]) < tol:
                indices.append(i)

            i += 1

        return indices

    @staticmethod
    def keep_observable_part(J, C_j, unobservable_indices):
        keep = [i for i in range(J.shape[0]) if i not in unobservable_indices]
        J_star = J[np.ix_(keep, keep)]
        C_star = C_j[:, keep]
        return J_star, C_star, keep

    @staticmethod
    def expand_observer_gain(L_star, keep, n):
        L_full = np.zeros((n, 1), dtype=complex)
        for local_idx, global_idx in enumerate(keep):
            L_full[global_idx, 0] = L_star[local_idx, 0]
        return L_full

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
        a = FourthSolver._sort_complex_list(spec1, tol)
        b = FourthSolver._sort_complex_list(spec2, tol)
        if len(a) != len(b):
            return False
        return all(abs(x - y) < tol for x, y in zip(a, b))

    @staticmethod
    def _format_spectrum(vals):
        return ", ".join(MatrixUtils.format_complex(v) for v in vals)

    @staticmethod
    def solve_order_1(A, C, desired_poles):
        lam_des = complex(desired_poles[0])
        a11 = complex(A[0, 0])
        c11 = complex(C[0, 0])

        if abs(c11) < 1e-12:
            raise ValueError(
                "для скалярной системы коэффициент C равен нулю, поэтому невозможно изменить собственное число наблюдателя"
            )

        l1 = (lam_des - a11) / c11
        L = np.array([[l1]], dtype=complex)
        Aobs = A + L @ C
        eig_obs = np.linalg.eigvals(Aobs)

        if not FourthSolver._spectra_match(eig_obs, [lam_des]):
            raise ValueError(
                "проверка не пройдена: после построения L спектр матрицы A + LC "
                "не совпал с желаемым спектром"
            )

        result = []
        result.append("Порядок системы равен 1, поэтому решаем задачу напрямую.")
        result.append("")
        result.append("Строим наблюдатель в виде")
        result.append("<pre>x̂̇ = Ax̂ + Bu + L(ŷ - y)</pre>")
        result.append("Тогда динамика ошибки задаётся матрицей")
        result.append("<pre>Aнабл = A + LC</pre>")
        result.append("")
        result.append("Матрица L:")
        result.append(f"<pre>{MatrixUtils.format_matrix(FourthSolver._cleanup_complex(L))}</pre>")
        result.append("")
        result.append("Матрица A + LC:")
        result.append(f"<pre>{MatrixUtils.format_matrix(FourthSolver._cleanup_complex(Aobs))}</pre>")
        result.append("")
        result.append("Проверка:")
        result.append("полученный спектр совпал с желаемым.")
        for lam in eig_obs:
            result.append(f"λ = {MatrixUtils.format_complex(lam)}")

        return L, Aobs, "\n".join(result)

    @staticmethod
    def solve_order_2_trace_det(A, C, desired_poles):
        if len(desired_poles) != 2:
            raise ValueError("для системы второго порядка нужно задать ровно 2 собственных числа")

        a11, a12 = complex(A[0, 0]), complex(A[0, 1])
        a21, a22 = complex(A[1, 0]), complex(A[1, 1])
        c1, c2 = complex(C[0, 0]), complex(C[0, 1])

        l1, l2 = sp.symbols("l1 l2")
        Aobs_sym = sp.Matrix([
            [sp.nsimplify(a11) + l1 * sp.nsimplify(c1), sp.nsimplify(a12) + l1 * sp.nsimplify(c2)],
            [sp.nsimplify(a21) + l2 * sp.nsimplify(c1), sp.nsimplify(a22) + l2 * sp.nsimplify(c2)],
        ])

        lam1 = sp.nsimplify(desired_poles[0])
        lam2 = sp.nsimplify(desired_poles[1])

        tr_eq = sp.Eq(Aobs_sym.trace(), sp.expand(lam1 + lam2))
        det_eq = sp.Eq(sp.expand(Aobs_sym.det()), sp.expand(lam1 * lam2))

        sol = sp.solve((tr_eq, det_eq), (l1, l2), dict=True)
        if not sol:
            raise ValueError(
                "не удалось решить систему по следу и определителю. "
                "Значит, для заданного спектра и этой пары (C, A) подходящий наблюдатель в таком виде не найден"
            )

        sol = sol[0]
        L = np.array([[complex(sp.N(sol[l1]))], [complex(sp.N(sol[l2]))]], dtype=complex)
        Aobs = A + L @ C
        eig_obs = np.linalg.eigvals(Aobs)

        if not FourthSolver._spectra_match(eig_obs, desired_poles):
            raise ValueError(
                "проверка не пройдена: после решения по следу и определителю "
                "спектр матрицы A + LC не совпал с желаемым"
            )

        s = complex(desired_poles[0] + desired_poles[1])
        p = complex(desired_poles[0] * desired_poles[1])
        disc = s * s - 4 * p

        result = []
        result.append("Порядок системы равен 2, поэтому решаем задачу через след и определитель.")
        result.append("")
        result.append("Строим наблюдатель в виде")
        result.append("<pre>x̂̇ = Ax̂ + Bu + L(ŷ - y)</pre>")
        result.append("")
        result.append("Тогда")
        result.append("<pre>Aнабл = A + LC</pre>")
        result.append("")
        result.append("Используем свойства:")
        result.append("<pre>tr(A + LC) = λ1 + λ2</pre>")
        result.append("<pre>det(A + LC) = λ1 λ2</pre>")
        result.append("")
        result.append("Дискриминант желаемого характеристического полинома:")
        result.append(f"<pre>D = (λ1 + λ2)^2 - 4 λ1 λ2 = {MatrixUtils.format_complex(disc)}</pre>")
        result.append("")
        result.append("Составляем систему уравнений:")
        result.append(f"<pre>{sp.sstr(sp.expand(tr_eq.lhs))} = {sp.sstr(sp.expand(tr_eq.rhs))}</pre>")
        result.append(f"<pre>{sp.sstr(sp.expand(det_eq.lhs))} = {sp.sstr(sp.expand(det_eq.rhs))}</pre>")
        result.append("")
        result.append("Матрица L:")
        result.append(f"<pre>{MatrixUtils.format_matrix(FourthSolver._cleanup_complex(L))}</pre>")
        result.append("")
        result.append("Матрица A + LC:")
        result.append(f"<pre>{MatrixUtils.format_matrix(FourthSolver._cleanup_complex(Aobs))}</pre>")
        result.append("")
        result.append("Проверка:")
        result.append("полученный спектр совпал с желаемым.")
        for lam in eig_obs:
            result.append(f"λ = {MatrixUtils.format_complex(lam)}")

        return L, Aobs, "\n".join(result)

    def solve_full_small_order(self, A, C, desired_poles):
        n = A.shape[0]
        if n == 1:
            return self.solve_order_1(A, C, desired_poles)
        if n == 2:
            return self.solve_order_2_trace_det(A, C, desired_poles)
        raise ValueError("метод для малых порядков применим только при n <= 2")

    def solve_full_sylvester(self, A, C, desired_poles):
        n = A.shape[0]

        if not self.is_observable(A, C):
            raise ValueError(
                "не выполнено условие метода Сильвестра из презентации: пара (C, A) не полностью наблюдаема"
            )

        if n <= 2:
            return self.solve_full_small_order(A, C, desired_poles)

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

        Y = self.choose_controllable_Y(Gamma)

        if not self.is_controllable(Gamma, Y):
            raise ValueError(
                "не выполнено условие метода Сильвестра: пара (Γ, Y) не управляема"
            )

        Q = self.solve_sylvester_numeric_observer(A, C, Gamma, Y)

        if not np.all(np.isfinite(Q)):
            raise ValueError(
                "при решении уравнения Сильвестра получены некорректные значения матрицы Q"
            )

        if np.linalg.matrix_rank(Q, self.TOL) < n:
            raise ValueError(
                "матрица Q оказалась вырожденной, поэтому невозможно вычислить L = -Q^-1Y"
            )

        L = -np.linalg.inv(Q) @ Y
        Aobs = A + L @ C
        eig_obs = np.linalg.eigvals(Aobs)

        if not self._spectra_match(eig_obs, desired_poles):
            raise ValueError(
                "проверка не пройдена: полученный спектр матрицы A + LC не совпал с желаемым. "
                f"Желаемый спектр: [{self._format_spectrum(desired_poles)}]. "
                f"Полученный спектр: [{self._format_spectrum(eig_obs)}]. "
                "Если все теоретические условия соблюдены, а проверка не проходит, "
                "значит проблема обычно связана с численной неточностью или неудачной постановкой Γ"
            )

        result = []
        result.append("Система полностью наблюдаема.")
        result.append("Так как порядок системы больше 2, синтезируем наблюдатель по матричному уравнению Сильвестра.")
        result.append("")
        result.append("Строим наблюдатель в виде")
        result.append("<pre>x̂̇ = Ax̂ + Bu + L(ŷ - y)</pre>")
        result.append("Тогда матрица динамики ошибки")
        result.append("<pre>Aнабл = A + LC</pre>")
        result.append("")
        result.append("Проверяем условия метода Сильвестра:")
        result.append("<pre>σ(A) ∩ σ(Γ) = ∅</pre>")
        result.append("<pre>(Γ, Y) — управляема</pre>")
        result.append("<pre>(C, A) — наблюдаема</pre>")
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
        result.append("<pre>QA - ΓQ = YC</pre>")
        result.append("и затем находим")
        result.append("<pre>L = -Q^-1Y</pre>")
        result.append("")
        result.append("Матрица Q:")
        result.append(f"<pre>{MatrixUtils.format_matrix(self._cleanup_complex(Q))}</pre>")
        result.append("")
        result.append("Матрица L:")
        result.append(f"<pre>{MatrixUtils.format_matrix(self._cleanup_complex(L))}</pre>")
        result.append("")
        result.append("Матрица A + LC:")
        result.append(f"<pre>{MatrixUtils.format_matrix(self._cleanup_complex(Aobs))}</pre>")
        result.append("")
        result.append("Проверка:")
        result.append("полученный спектр совпал с желаемым.")
        for lam in eig_obs:
            result.append(f"λ = {MatrixUtils.format_complex(lam)}")

        return L, Aobs, "\n".join(result)

    def solve_with_truncation(self, A, C, desired_poles):
        n = A.shape[0]

        J, P = self.jordan_decomposition_numeric(A)
        P_inv = np.linalg.inv(P)
        C_j = C @ P

        unobservable_indices = self.find_unobservable_indices_in_jordan(J, C_j)

        if len(unobservable_indices) == 0:
            return self.solve_full_sylvester(A, C, desired_poles)

        for idx in unobservable_indices:
            lam = J[idx, idx]
            if np.real(lam) >= 0:
                raise ValueError(
                    "система не обнаруживаема: среди ненаблюдаемых мод есть неустойчивые собственные числа"
                )

        J_star, C_star, keep = self.keep_observable_part(J, C_j, unobservable_indices)
        r = J_star.shape[0]

        if len(desired_poles) != r:
            raise ValueError(
                f"после усечения порядок наблюдаемой подсистемы равен {r}, "
                f"поэтому нужно задать ровно {r} собственных чисел"
            )

        if not self.is_observable(J_star, C_star):
            raise ValueError(
                "после усечения подсистема должна быть полностью наблюдаема, но проверка наблюдаемости не пройдена"
            )

        result = []
        result.append("Система не полностью наблюдаема, но обнаруживаема.")
        result.append("Поэтому выполняем синтез через усечение.")
        result.append("")
        result.append("1) Переходим к жордановой форме в координатах")
        result.append("<pre>x^ = P^-1 x</pre>")
        result.append("<pre>x = P x^</pre>")
        result.append("Тогда")
        result.append("<pre>x^dot = AJ x^ + BJ u</pre>")
        result.append("<pre>y = CJ x^ + Du</pre>")
        result.append("<pre>AJ = P^-1 A P</pre>")
        result.append("<pre>CJ = C P</pre>")
        result.append("")
        result.append("2) Вычёркиваем ненаблюдаемые устойчивые моды.")
        result.append("3) Получаем наблюдаемую подсистему меньшего порядка.")
        result.append("")
        result.append("Матрица AJ:")
        result.append(f"<pre>{MatrixUtils.format_matrix(self._cleanup_complex(J))}</pre>")
        result.append("")
        result.append("Матрица CJ:")
        result.append(f"<pre>{MatrixUtils.format_matrix(self._cleanup_complex(C_j))}</pre>")
        result.append("")
        result.append("Матрица AJ*:")
        result.append(f"<pre>{MatrixUtils.format_matrix(self._cleanup_complex(J_star))}</pre>")
        result.append("")
        result.append("Матрица CJ*:")
        result.append(f"<pre>{MatrixUtils.format_matrix(self._cleanup_complex(C_star))}</pre>")
        result.append("")

        if r <= 2:
            L_j_star, _, small_text = self.solve_full_small_order(J_star, C_star, desired_poles)
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

            Y_star = self.choose_controllable_Y(Gamma_star)

            if not self.is_controllable(Gamma_star, Y_star):
                raise ValueError(
                    "для усечённой подсистемы не выполнено условие управляемости пары (Γ*, Y*)"
                )

            Q_star = self.solve_sylvester_numeric_observer(J_star, C_star, Gamma_star, Y_star)

            if not np.all(np.isfinite(Q_star)):
                raise ValueError(
                    "для усечённой подсистемы при решении Сильвестра получены некорректные значения Q*"
                )

            if np.linalg.matrix_rank(Q_star, self.TOL) < r:
                raise ValueError(
                    "после усечения матрица Q* вырождена, поэтому невозможно вычислить LJ* = -Q*^-1Y*"
                )

            L_j_star = -np.linalg.inv(Q_star) @ Y_star
            Aobs_star = J_star + L_j_star @ C_star
            eig_star = np.linalg.eigvals(Aobs_star)

            if not self._spectra_match(eig_star, desired_poles):
                raise ValueError(
                    "для усечённой подсистемы проверка не пройдена: "
                    "полученный спектр не совпал с желаемым"
                )

            result.append("Для усечённой подсистемы применяем Сильвестра.")
            result.append("Проверки выполнены:")
            result.append("<pre>σ(AJ*) ∩ σ(Γ*) = ∅</pre>")
            result.append("<pre>(Γ*, Y*) — управляема</pre>")
            result.append("<pre>(CJ*, AJ*) — наблюдаема</pre>")
            result.append("")
            result.append("Матрица Γ*:")
            result.append(f"<pre>{MatrixUtils.format_matrix(self._cleanup_complex(Gamma_star))}</pre>")
            result.append("")
            result.append("Матрица Y*:")
            result.append(f"<pre>{MatrixUtils.format_matrix(self._cleanup_complex(Y_star))}</pre>")
            result.append("")
            result.append("Матрица Q*:")
            result.append(f"<pre>{MatrixUtils.format_matrix(self._cleanup_complex(Q_star))}</pre>")
            result.append("")
            result.append("Наблюдатель для усечённой подсистемы LJ*:")
            result.append(f"<pre>{MatrixUtils.format_matrix(self._cleanup_complex(L_j_star))}</pre>")
            result.append("")

        L_j = self.expand_observer_gain(L_j_star, keep, n)
        L = P @ L_j
        Aobs = A + L @ C
        eig_obs = np.linalg.eigvals(Aobs)

        result.append("Дополняем наблюдатель нулями до LJ.")
        result.append("Так как")
        result.append("<pre>LJ = P^-1 L</pre>")
        result.append("то в исходных координатах")
        result.append("<pre>L = P LJ</pre>")
        result.append("")
        result.append("Матрица L:")
        result.append(f"<pre>{MatrixUtils.format_matrix(self._cleanup_complex(L))}</pre>")
        result.append("")
        result.append("Матрица A + LC:")
        result.append(f"<pre>{MatrixUtils.format_matrix(self._cleanup_complex(Aobs))}</pre>")
        result.append("")
        result.append("Собственные числа наблюдателя:")
        for lam in eig_obs:
            result.append(f"λ = {MatrixUtils.format_complex(lam)}")

        return L, Aobs, "\n".join(result)

    def reduced_order_for_fourth_task(self, A, C):
        if self.is_observable(A, C):
            return A.shape[0]

        J, P = self.jordan_decomposition_numeric(A)
        C_j = C @ P
        unobservable_indices = self.find_unobservable_indices_in_jordan(J, C_j)
        keep = [i for i in range(A.shape[0]) if i not in unobservable_indices]
        return len(keep)

    def solve(self, A, C, poles_input=None):
        n = A.shape[0]

        if A.shape[0] != A.shape[1]:
            return "Ошибка: матрица A должна быть квадратной."

        if C.shape[1] != n:
            return "Ошибка: число столбцов матрицы C должно совпадать с размерностью матрицы A."

        if C.shape[0] != 1:
            return "Ошибка: в задании 4 матрица C должна быть строкой."

        try:
            detectable = self.is_detectable(A, C)
        except Exception as e:
            return f"Ошибка при анализе обнаруживаемости: {e}"

        if not detectable:
            return (
                "Решение задания 4\n\n"
                "Синтез наблюдателя невозможен.\n"
                "Причина: система не обнаруживаема."
            )

        try:
            if poles_input and str(poles_input).strip():
                desired_poles = self.parse_poles(poles_input)
            else:
                target_order = n if self.is_observable(A, C) else self.reduced_order_for_fourth_task(A, C)
                desired_poles = self.default_observer_poles(target_order)
        except Exception as e:
            return (
                "Решение задания 4\n\n"
                "Не удалось разобрать желаемый спектр.\n"
                f"Причина: {e}"
            )

        try:
            result = ["Решение задания 4", ""]

            if self.is_observable(A, C):
                if len(desired_poles) != n:
                    return (
                        "Решение задания 4\n\n"
                        f"Ошибка: для системы порядка {n} нужно задать ровно {n} собственных чисел."
                    )
                _, _, text = self.solve_full_sylvester(A, C, desired_poles)
                result.append(text)
            else:
                _, _, text = self.solve_with_truncation(A, C, desired_poles)
                result.append(text)

            return "\n".join(result)

        except Exception as e:
            return (
                "Решение задания 4\n\n"
                "Не удалось построить наблюдатель.\n"
                f"Причина: {e}"
            )