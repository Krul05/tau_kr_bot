from matrix_utils import MatrixUtils
import sympy as sym


class ThirdSolver:
    @staticmethod
    def _charpoly_data(A_sym):
        lam = sym.Symbol('λ')
        expr = sym.expand((A_sym - lam * sym.eye(A_sym.rows)).det())
        roots = A_sym.eigenvals()
        return lam, expr, roots

    @staticmethod
    def _solve_generalized_chain(A_sym, lam, start_vector, block_size):
        """
        Строит одну жорданову цепочку для λ:
        v1 - собственный вектор,
        далее (A - λI)v_k = v_{k-1}
        """
        n = A_sym.rows
        M = A_sym - lam * sym.eye(n)

        chain_vectors = [sym.Matrix(start_vector)]
        equations_text = []

        prev = sym.Matrix(start_vector)
        for k in range(2, block_size + 1):
            sol = sym.linsolve((M, prev))
            if not sol:
                break

            vk = sym.Matrix(list(sol)[0])
            chain_vectors.append(vk)
            equations_text.append({
                "step": k,
                "equation": f"(A - {MatrixUtils.format_sympy_number(lam)}I)w_{k-1} = "
                            f"{'v_1' if k == 2 else f'w_{k-2}'}",
                "vector": vk
            })
            prev = vk

        return chain_vectors, equations_text

    @staticmethod
    def _find_real_jordan_chains(A_sym):
        """
        Строит вещественный базис для вещественной жордановой формы.
        Если λ вещественное -> обычные жордановы цепочки.
        Если λ = a+bi, b>0 -> берём цепочки для λ и заменяем их на Re/Im части.
        """
        n = A_sym.rows
        eig_data = A_sym.eigenvects()
        _, jordan_cells = A_sym.jordan_cells()

        chains_info = []
        p_columns = []
        real_blocks = []

        processed_complex_keys = set()

        for eig_index, item in enumerate(eig_data, start=1):
            lam = sym.simplify(item[0])
            algebraic_mult = item[1]
            eig_basis = [sym.Matrix(v) for v in item[2]]

            M = A_sym - lam * sym.eye(n)
            rref_M, pivots = M.rref()

            relevant_cells = [cell for cell in jordan_cells if sym.simplify(cell[0, 0]) == lam]
            block_sizes = [cell.rows for cell in relevant_cells]

            if MatrixUtils.is_real_eigenvalue(lam):
                used_eigenvectors = 0
                local_chains = []

                for block_size in block_sizes:
                    if used_eigenvectors >= len(eig_basis):
                        break

                    v1 = eig_basis[used_eigenvectors]
                    used_eigenvectors += 1

                    chain_vectors, equations_text = ThirdSolver._solve_generalized_chain(
                        A_sym, lam, v1, block_size
                    )

                    local_chains.append({
                        "block_size": block_size,
                        "lambda": lam,
                        "is_complex_pair": False,
                        "eigenvector": v1,
                        "chain_vectors": chain_vectors,
                        "equations": equations_text
                    })

                    for vec in chain_vectors:
                        p_columns.append(vec)

                    real_blocks.append({
                        "lambda": lam,
                        "size": block_size,
                        "is_complex_pair": False,
                        "real_block_size": block_size
                    })

                left_names, right_exprs = MatrixUtils.eigenvector_parametric_formula(rref_M, eig_index)

                chains_info.append({
                    "eigen_index": eig_index,
                    "lambda": lam,
                    "algebraic_mult": algebraic_mult,
                    "geometric_mult": len(eig_basis),
                    "matrix_A_minus_lambdaI": M,
                    "rref_matrix": rref_M,
                    "pivots": pivots,
                    "eigenvectors": eig_basis,
                    "chains": local_chains,
                    "block_sizes": block_sizes,
                    "parametric_left_names": left_names,
                    "parametric_right_exprs": right_exprs,
                    "used_real_jordan_form": False
                })

            else:
                key = (sym.re(lam), abs(sym.im(lam)))
                if key in processed_complex_keys:
                    continue
                if not MatrixUtils.is_positive_imag_eigenvalue(lam):
                    continue

                processed_complex_keys.add(key)

                used_eigenvectors = 0
                local_chains = []

                for block_size in block_sizes:
                    if used_eigenvectors >= len(eig_basis):
                        break

                    v1 = eig_basis[used_eigenvectors]
                    used_eigenvectors += 1

                    chain_vectors, equations_text = ThirdSolver._solve_generalized_chain(
                        A_sym, lam, v1, block_size
                    )

                    real_chain_vectors = []
                    for vec in chain_vectors:
                        re_vec = MatrixUtils.real_part_vector(vec)
                        im_vec = MatrixUtils.imag_part_vector(vec)
                        real_chain_vectors.append(re_vec)
                        real_chain_vectors.append(im_vec)
                        p_columns.append(re_vec)
                        p_columns.append(im_vec)

                    local_chains.append({
                        "block_size": block_size,
                        "lambda": lam,
                        "is_complex_pair": True,
                        "eigenvector": v1,
                        "chain_vectors": chain_vectors,
                        "real_chain_vectors": real_chain_vectors,
                        "equations": equations_text
                    })

                    real_blocks.append({
                        "lambda": lam,
                        "size": block_size,
                        "is_complex_pair": True,
                        "real_block_size": 2 * block_size
                    })

                left_names = [f"v_{{{eig_index}{j+1}}}" for j in range(n)]
                right_exprs = [MatrixUtils.format_sympy_number(eig_basis[0][j]) for j in range(n)] if eig_basis else []

                chains_info.append({
                    "eigen_index": eig_index,
                    "lambda": lam,
                    "algebraic_mult": algebraic_mult,
                    "geometric_mult": len(eig_basis),
                    "matrix_A_minus_lambdaI": M,
                    "rref_matrix": rref_M,
                    "pivots": pivots,
                    "eigenvectors": eig_basis,
                    "chains": local_chains,
                    "block_sizes": block_sizes,
                    "parametric_left_names": left_names,
                    "parametric_right_exprs": right_exprs,
                    "used_real_jordan_form": True
                })

        if p_columns:
            P = sym.Matrix.hstack(*p_columns)
        else:
            P = sym.Matrix.zeros(A_sym.rows, 0)

        J_blocks = []
        for block in real_blocks:
            J_blocks.append(MatrixUtils.real_jordan_block(block["lambda"], block["size"]))

        if J_blocks:
            J_real = sym.diag(*J_blocks)
        else:
            J_real = sym.Matrix.zeros(A_sym.rows, A_sym.cols)

        return chains_info, P, J_real, real_blocks

    @staticmethod
    def _get_real_jordan_blocks_from_structure(real_blocks):
        blocks = []
        start = 0

        for idx, rb in enumerate(real_blocks, start=1):
            size = rb["real_block_size"]
            end = start + size - 1
            blocks.append({
                "index": idx,
                "lambda": rb["lambda"],
                "size": size,
                "start": start,
                "end": end,
                "is_complex_pair": rb["is_complex_pair"],
                "chain_size": rb["size"]
            })
            start = end + 1

        return blocks

    @staticmethod
    def jordan_observability_analysis(A, C):
        A_sym = MatrixUtils.to_sympy_matrix(A)
        C_sym = MatrixUtils.to_sympy_matrix(C)

        lam, char_poly, roots = ThirdSolver._charpoly_data(A_sym)
        chains_info, P, J_declared, real_blocks = ThirdSolver._find_real_jordan_chains(A_sym)

        P_inv = sym.simplify(P.inv())
        J = sym.simplify(P_inv * A_sym * P)
        H = sym.simplify(C_sym * P)

        blocks_structure = ThirdSolver._get_real_jordan_blocks_from_structure(real_blocks)

        block_results = []
        unobservable_basis_cols = []

        for block in blocks_structure:
            start = block["start"]
            end = block["end"]
            size = block["size"]

            h_block = H[:, start:end + 1]

            if not block["is_complex_pair"]:
                test_value = sym.simplify(h_block[0, 0])
                observable = (test_value != 0)
                criterion_text = f"первый элемент = {MatrixUtils.format_sympy_number(test_value)}"
            else:
                # Для вещественного блока комплексной пары проверяем первую пару элементов
                v1 = sym.simplify(h_block[0, 0])
                v2 = sym.simplify(h_block[0, 1])
                observable = not (v1 == 0 and v2 == 0)
                criterion_text = (
                    "первая пара элементов = "
                    f"({MatrixUtils.format_sympy_number(v1)}, {MatrixUtils.format_sympy_number(v2)})"
                )

            block_results.append({
                "index": block["index"],
                "lambda": block["lambda"],
                "size": size,
                "start": start,
                "end": end,
                "h_block": h_block,
                "observable": observable,
                "is_complex_pair": block["is_complex_pair"],
                "criterion_text": criterion_text
            })

            if not observable:
                for col in range(start, end + 1):
                    unobservable_basis_cols.append(P[:, col])

        eigen_results = []
        grouped = {}
        for br in block_results:
            key = sym.sstr(sym.simplify(br["lambda"]))
            if key not in grouped:
                grouped[key] = {"lambda": br["lambda"], "blocks": []}
            grouped[key]["blocks"].append(br)

        detectable = True
        for _, group in grouped.items():
            lam_value = group["lambda"]
            eig_observable = all(br["observable"] for br in group["blocks"])

            lam_complex = complex(sym.N(lam_value))
            if lam_complex.real >= -1e-10 and not eig_observable:
                detectable = False

            eigen_results.append({
                "lambda": lam_value,
                "observable": eig_observable,
                "blocks": group["blocks"]
            })

        if unobservable_basis_cols:
            unobservable_basis = sym.Matrix.hstack(*unobservable_basis_cols)
        else:
            unobservable_basis = sym.Matrix.zeros(A_sym.rows, 0)

        return {
            "A_sym": A_sym,
            "C_sym": C_sym,
            "lam": lam,
            "char_poly": char_poly,
            "roots": roots,
            "chains_info": chains_info,
            "P": P,
            "P_inv": P_inv,
            "J": J,
            "J_declared": J_declared,
            "H": H,
            "blocks": block_results,
            "eigen_results": eigen_results,
            "unobservable_basis": unobservable_basis,
            "detectable": detectable
        }

    def solve(self, A, C):
        n = A.shape[0]

        if A.shape[0] != A.shape[1]:
            return "Ошибка: матрица A должна быть квадратной."

        if C.shape[1] != n:
            return "Ошибка: число столбцов матрицы C должно совпадать с размерностью матрицы A."

        if C.shape[0] != 1:
            return "Ошибка: в задании 3 матрица C должна быть строкой."

        try:
            analysis = self.jordan_observability_analysis(A, C)
        except Exception as e:
            return f"Ошибка при построении жордановой формы: {e}"

        A_sym = analysis["A_sym"]
        C_sym = analysis["C_sym"]
        lam = analysis["lam"]
        char_poly = analysis["char_poly"]
        roots = analysis["roots"]
        chains_info = analysis["chains_info"]
        P = analysis["P"]
        P_inv = analysis["P_inv"]
        J = analysis["J"]
        H = analysis["H"]
        blocks = analysis["blocks"]
        eigen_results = analysis["eigen_results"]
        unobservable_basis = analysis["unobservable_basis"]
        detectable = analysis["detectable"]

        result = []

        result.append("Решение задания 3")
        result.append("")
        result.append("Задана система")
        result.append("<pre>")
        result.append("A =")
        result.append(MatrixUtils.format_sympy_matrix(A_sym))
        result.append("")
        result.append("C =")
        result.append(MatrixUtils.format_sympy_matrix(C_sym))
        result.append("</pre>")
        result.append("")

        result.append("Найдём собственные числа матрицы A.")
        result.append("Для этого составим характеристическую матрицу A - λI:")
        result.append(f"<pre>{MatrixUtils.format_sympy_matrix(A_sym - lam * sym.eye(A_sym.rows))}</pre>")

        result.append("Характеристическое уравнение:")
        result.append("<pre>det(A - λI) = 0</pre>")
        result.append(f"<pre>{sym.expand(char_poly)} = 0</pre>")

        eigen_list = []
        for eig, mult in roots.items():
            eig_str = MatrixUtils.format_sympy_number(eig)
            if mult == 1:
                eigen_list.append(f"λ = {eig_str}")
            else:
                eigen_list.append(f"λ = {eig_str}, кратность {mult}")

        result.append("Собственные числа:")
        result.append("<pre>" + ";\n".join(eigen_list) + "</pre>")
        result.append("")

        result.append("Найдём собственные векторы из уравнения")
        result.append("<pre>(A - λI)v = 0</pre>")
        result.append("и при необходимости присоединённые векторы из уравнений")
        result.append("<pre>(A - λI)w₁ = v,\n(A - λI)w₂ = w₁, ...</pre>")
        result.append("")

        for info in chains_info:
            lam_value = info["lambda"]
            lam_str = MatrixUtils.format_sympy_number(lam_value)

            result.append(f"Для собственного числа λ = {lam_str}:")
            result.append("Матрица A - λI:")
            result.append(f"<pre>{MatrixUtils.format_sympy_matrix(info['matrix_A_minus_lambdaI'])}</pre>")

            result.append("После приведения к ступенчатому виду получаем:")
            result.append(f"<pre>{MatrixUtils.format_sympy_matrix(info['rref_matrix'])}</pre>")

            if MatrixUtils.is_real_eigenvalue(lam_value):
                result.append("Параметрический вид собственного вектора:")
                result.append(
                    "<pre>" +
                    MatrixUtils.format_parametric_vector_formula(
                        info["parametric_left_names"],
                        info["parametric_right_exprs"]
                    ) +
                    "</pre>"
                )
            else:
                result.append("Собственное число комплексное, поэтому далее строим вещественную жорданову форму.")
                result.append("Берём комплексную жорданову цепочку и заменяем её на вещественные и мнимые части векторов.")

            result.append(f"Алгебраическая кратность: {info['algebraic_mult']}")
            result.append(f"Геометрическая кратность: {info['geometric_mult']}")

            result.append("Собственные векторы:")
            for idx, vec in enumerate(info["eigenvectors"], start=1):
                result.append(f"v{idx} =")
                result.append(f"<pre>{MatrixUtils.format_sympy_matrix(vec)}</pre>")

            if info["chains"]:
                result.append("Жордановы цепочки:")
                for chain_index, chain in enumerate(info["chains"], start=1):
                    result.append(f"Цепочка #{chain_index}, размер блока {chain['block_size']}")
                    result.append("Собственный вектор:")
                    result.append(f"<pre>{MatrixUtils.format_sympy_matrix(chain['eigenvector'])}</pre>")

                    if chain["equations"]:
                        result.append("Присоединённые векторы:")
                        for eq_info in chain["equations"]:
                            result.append(eq_info["equation"])
                            result.append(f"<pre>{MatrixUtils.format_sympy_matrix(eq_info['vector'])}</pre>")
                    else:
                        result.append("Присоединённые векторы не требуются, так как размер блока равен 1.")

                    if chain.get("is_complex_pair", False):
                        result.append("Переход к вещественной жордановой цепочке:")
                        for idx_vec, rv in enumerate(chain["real_chain_vectors"], start=1):
                            label = "Re" if idx_vec % 2 == 1 else "Im"
                            result.append(f"{label}-часть:")
                            result.append(f"<pre>{MatrixUtils.format_sympy_matrix(rv)}</pre>")
            result.append("")

        result.append("Составим матрицу перехода P из найденных векторов.")
        result.append("Если есть комплексно-сопряжённые собственные числа,")
        result.append("то в P включаются вещественные и мнимые части соответствующих цепочек.")
        result.append(f"<pre>P =\n{MatrixUtils.format_sympy_matrix(P)}</pre>")

        result.append("Найдём обратную матрицу:")
        result.append(f"<pre>P^(-1) =\n{MatrixUtils.format_sympy_matrix(P_inv)}</pre>")

        result.append("Тогда вещественная жорданова форма равна")
        result.append("<pre>J = P^(-1) A P</pre>")
        result.append(f"<pre>J =\n{MatrixUtils.format_sympy_matrix(J)}</pre>")

        result.append("Для уравнения выхода переходим к координатам x = Pz.")
        result.append("Тогда")
        result.append("<pre>y = Cx = CPz = Hz</pre>")
        result.append("где")
        result.append("<pre>H = CP</pre>")
        result.append(f"<pre>H =\n{MatrixUtils.format_sympy_matrix(H)}</pre>")
        result.append("")

        result.append("Дальше исследуем наблюдаемость блоков вещественной жордановой формы.")
        result.append("Для вещественного блока проверяется первый элемент соответствующего фрагмента H.")
        result.append("Для блока, соответствующего комплексно-сопряжённой паре,")
        result.append("проверяется первая пара элементов: они не должны одновременно обращаться в нуль.")
        result.append("")

        for br in blocks:
            lam_str = MatrixUtils.format_sympy_number(br["lambda"])
            status = "наблюдаем" if br["observable"] else "ненаблюдаем"

            result.append(f"Блок #{br['index']}: λ = {lam_str}, размер = {br['size']}")
            if br["is_complex_pair"]:
                result.append("Это блок вещественной жордановой формы для комплексно-сопряжённой пары.")
            result.append("Соответствующий фрагмент H:")
            result.append(f"<pre>{MatrixUtils.format_sympy_matrix(br['h_block'])}</pre>")
            result.append(br["criterion_text"])
            result.append(f"Следовательно, блок {status}.")
            result.append("")

        result.append("Вывод по собственным числам:")
        for er in eigen_results:
            lam_str = MatrixUtils.format_sympy_number(er["lambda"])
            status = "наблюдаемо" if er["observable"] else "ненаблюдаемо"
            result.append(f"λ = {lam_str} → {status}")

        result.append("")
        result.append(f"Ответ: {'система обнаруживаема' if detectable else 'система не обнаруживаема'}.")

        if detectable:
            result.append("Все собственные числа с неотрицательной вещественной частью наблюдаемы.")
        else:
            result.append("Существует ненаблюдаемый блок, соответствующий собственному числу")
            result.append("с неотрицательной вещественной частью.")

        result.append("")
        result.append("Ненаблюдаемое подпространство:")
        if unobservable_basis.cols == 0:
            result.append("Оно тривиально и состоит только из нулевого вектора.")
        else:
            result.append("Базис ненаблюдаемого подпространства:")
            result.append(f"<pre>{MatrixUtils.format_sympy_matrix(unobservable_basis)}</pre>")
            result.append(f"Размерность: {unobservable_basis.cols}")

        return "\n".join(result)