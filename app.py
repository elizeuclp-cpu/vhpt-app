import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import pandas as pd
from datetime import datetime

# ===================================================
# CONFIGURAÇÃO DA PÁGINA
# ===================================================
st.set_page_config(
    page_title="VHPT - Verificador de Hipóteses para Postes",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================================================
# BANCO DE DADOS DOS CABOS CONDUTORES
# ===================================================
cabos_data = {
    1: {'nome': 'Grosbeak', 'peso': 1.3028, 'area': 374.8, 'E': 7593, 'alpha': 189e-7, 'CR': 11427, 'D': 0.02515},
    2: {'nome': 'Linnet', 'peso': 0.6883, 'area': 198, 'E': 7593, 'alpha': 189e-7, 'CR': 6393, 'D': 0.0183},
    3: {'nome': 'Penguin', 'peso': 0.433, 'area': 125.06, 'E': 8120, 'alpha': 186e-7, 'CR': 3790, 'D': 0.01431},
    4: {'nome': 'Cairo AAAC 6201', 'peso': 0.651, 'area': 236.38, 'E': 8120, 'alpha': 186e-7, 'CR': 7106.3, 'D': 0.0199},
    5: {'nome': 'Raven', 'peso': 0.217, 'area': 125.09, 'E': 7593, 'alpha': 189e-7, 'CR': 1985, 'D': 0.01011},
    6: {'nome': 'Flint', 'peso': 1.035, 'area': 374.52, 'E': 8120, 'alpha': 186e-7, 'CR': 11012.8, 'D': 0.0251},
    7: {'nome': 'Drake', 'peso': 1.629, 'area': 468, 'E': 7593, 'alpha': 189e-7, 'CR': 14245, 'D': 0.02813}
}

# ===================================================
# BANCO DE DADOS DOS CABOS PARA-RAIOS
# ===================================================
cabos_pr_data = {
    'Tipo A': {'nome': 'Para-raios Tipo A', 'CR': 5000, 'peso': 0.5, 'E': 10000, 'alpha': 180e-7, 'D': 0.012},
    'Tipo B': {'nome': 'Para-raios Tipo B', 'CR': 7000, 'peso': 0.7, 'E': 12000, 'alpha': 185e-7, 'D': 0.014},
    'Tipo C': {'nome': 'Para-raios Tipo C', 'CR': 10000, 'peso': 1.0, 'E': 15000, 'alpha': 190e-7, 'D': 0.016}
}

# ===================================================
# CONSTANTES
# ===================================================
KGf_TO_DAN = 0.981
SOBRECARGA_TRANSITORIA = 0.40

# ===================================================
# FUNÇÕES AUXILIARES
# ===================================================

def mudanca_estado(T_initial, temp_initial, temp_final, peso_initial, peso_final, vao, E, S, alpha):
    B = (E * S * peso_initial**2 * vao**2) / (24 * T_initial**2) + E * S * alpha * (temp_final - temp_initial) - T_initial
    C = (E * S * peso_final**2 * vao**2) / 24
    roots = np.roots([1, B, 0, -C])
    T_final = roots[np.isreal(roots) & (roots > 0)].real
    if len(T_final) == 0:
        return T_initial
    return T_final[0]

def calcular_coeficientes_k(dist_fases_topo, dist_pr_topo, altura_poste):
    enterrado = 0.10 * altura_poste + 0.60
    parte_aerea = altura_poste - enterrado
    X = parte_aerea - 0.20
    Y_fases = [parte_aerea - dist for dist in dist_fases_topo]
    K_fases = [Y / X for Y in Y_fases]
    if dist_pr_topo is not None and dist_pr_topo > 0:
        Y_pr = parte_aerea - dist_pr_topo
        K_pr = Y_pr / X
    else:
        K_pr = 0
    return K_fases, K_pr, X, parte_aerea

def calcular_tracoes_eds(cr, perc_re, perc_vante, tipo_fixacao):
    T_re = cr * (perc_re / 100)
    if tipo_fixacao == 'suspensao':
        T_vante = T_re
    else:
        T_vante = cr * (perc_vante / 100)
    return T_re, T_vante

def forca_complexa(magnitude, angulo_graus):
    return magnitude * np.exp(1j * np.radians(angulo_graus))

def fator_correcao_duplo_t(theta_res, theta_face_lisa):
    delta_theta = abs(theta_res - theta_face_lisa)
    if delta_theta > 90:
        delta_theta = 180 - delta_theta
    rad = np.radians(delta_theta)
    a, b = 1.0, 0.5
    return 1 / np.sqrt((np.cos(rad)/a)**2 + (np.sin(rad)/b)**2)

def encontrar_angulo_critico_vento(F_re, F_vante, ang_re, ang_vante, S_arrasto, P_vento, L_re, L_vante):
    melhor_angulo = 90
    maior_resultante = 0
    F_cabos = forca_complexa(F_re, ang_re) + forca_complexa(F_vante, ang_vante)
    
    for ang_vento in range(90, 181, 1):
        seno_re = abs(np.sin(np.radians(ang_vento - ang_re)))
        seno_vante = abs(np.sin(np.radians(ang_vento - ang_vante)))
        F_vento_mag = (P_vento/2) * (L_re * seno_re + L_vante * seno_vante) * S_arrasto
        F_vento = forca_complexa(F_vento_mag, ang_vento)
        R_total = abs(F_cabos + F_vento)
        if R_total > maior_resultante:
            maior_resultante = R_total
            melhor_angulo = ang_vento
    
    return melhor_angulo, maior_resultante

def plotar_diagrama(F_re, F_vante, F_vento, F_res, ang_re, ang_vante, ang_vento, theta_res, 
                    tipo_poste, titulo, theta_face_lisa=None, fator_correcao=1.0):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    max_force = max(abs(F_re), abs(F_vante), abs(F_vento), abs(F_res))
    scale = max_force / 1.2 if max_force > 0 else 1
    
    ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)
    
    if max_force > 0:
        circle = plt.Circle((0, 0), max_force/scale, fill=False, linestyle='--', alpha=0.2, color='gray')
        ax.add_patch(circle)
    
    if F_re > 0:
        x_re = F_re * np.cos(np.radians(ang_re)) / scale
        y_re = F_re * np.sin(np.radians(ang_re)) / scale
        ax.arrow(0, 0, x_re, y_re, head_width=0.08, head_length=0.08,
                 fc='red', ec='red', linewidth=2.5, label=f'Vão Ré: {F_re:.0f} kgf')
    
    if F_vante > 0:
        x_vante = F_vante * np.cos(np.radians(ang_vante)) / scale
        y_vante = F_vante * np.sin(np.radians(ang_vante)) / scale
        ax.arrow(0, 0, x_vante, y_vante, head_width=0.08, head_length=0.08,
                 fc='blue', ec='blue', linewidth=2.5, label=f'Vão Vante: {F_vante:.0f} kgf')
    
    if F_vento > 0:
        x_vento = F_vento * np.cos(np.radians(ang_vento)) / scale
        y_vento = F_vento * np.sin(np.radians(ang_vento)) / scale
        ax.arrow(0, 0, x_vento, y_vento, head_width=0.08, head_length=0.08,
                 fc='orange', ec='orange', linewidth=2.5, label=f'Vento: {F_vento:.0f} kgf')
    
    if F_res > 0:
        x_res = F_res * np.cos(np.radians(theta_res)) / scale
        y_res = F_res * np.sin(np.radians(theta_res)) / scale
        ax.arrow(0, 0, x_res, y_res, head_width=0.1, head_length=0.1,
                 fc='green', ec='green', linewidth=3.5, label=f'Resultante: {F_res:.0f} kgf')
    
    if tipo_poste == 'Duplo T' and theta_face_lisa is not None:
        theta_gaveta = theta_face_lisa + 90
        
        x_lisa = 1.2 * np.cos(np.radians(theta_face_lisa))
        y_lisa = 1.2 * np.sin(np.radians(theta_face_lisa))
        ax.plot([-x_lisa, x_lisa], [-y_lisa, y_lisa], 'purple', linewidth=2, linestyle='--', alpha=0.8)
        ax.text(x_lisa*1.05, y_lisa*1.05, 'Face Lisa', color='purple', fontsize=10, alpha=0.8, fontweight='bold')
        
        x_gaveta = 1.2 * np.cos(np.radians(theta_gaveta))
        y_gaveta = 1.2 * np.sin(np.radians(theta_gaveta))
        ax.plot([-x_gaveta, x_gaveta], [-y_gaveta, y_gaveta], 'orange', linewidth=2, linestyle='--', alpha=0.8)
        ax.text(x_gaveta*1.05, y_gaveta*1.05, 'Gaveta', color='orange', fontsize=10, alpha=0.8, fontweight='bold')
        
        if max_force > 0:
            a = 1.0 * (max_force/scale) * 0.6
            b = 0.5 * (max_force/scale) * 0.6
            ellipse = Ellipse((0, 0), 2*a, 2*b, angle=theta_face_lisa,
                              facecolor='none', edgecolor='gray', linestyle=':', linewidth=2, alpha=0.7)
            ax.add_patch(ellipse)
            
            rad_res = np.radians(theta_res - theta_face_lisa)
            r_ellipse = (a * b) / np.sqrt((b * np.cos(rad_res))**2 + (a * np.sin(rad_res))**2)
            x_ellipse_point = r_ellipse * np.cos(np.radians(theta_res))
            y_ellipse_point = r_ellipse * np.sin(np.radians(theta_res))
            ax.plot(x_ellipse_point, y_ellipse_point, 'ko', markersize=8, zorder=5)
            ax.annotate(f'Fator: {fator_correcao:.3f}',
                       xy=(x_ellipse_point, y_ellipse_point),
                       xytext=(x_ellipse_point*1.1, y_ellipse_point*1.1),
                       fontsize=8, color='black')
    
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Eixo X', fontsize=12)
    ax.set_ylabel('Eixo Y', fontsize=12)
    ax.set_title(titulo, fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    
    ax.text(1.05, 0, '0°', fontsize=8, alpha=0.7)
    ax.text(0, 1.05, '90°', fontsize=8, alpha=0.7)
    ax.text(-1.1, 0, '180°', fontsize=8, alpha=0.7)
    ax.text(0, -1.1, '270°', fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    return fig

# ===================================================
# INTERFACE STREAMLIT
# ===================================================

st.title("🏗️ VHPT - VERIFICADOR DE HIPÓTESES PARA POSTES")
st.markdown("---")

# Criar colunas para layout
col1, col2 = st.columns([1, 1], gap="medium")

with col1:
    st.subheader("🏗️ 1. CONFIGURAÇÃO DO POSTE")
    
    tipo_poste = st.selectbox("Tipo de poste", ["Circular", "Duplo T"], index=1)
    fixacao = st.selectbox("Tipo de fixação", ["suspensao", "ancoragem"], index=1, format_func=lambda x: "Suspensão" if x == "suspensao" else "Ancoragem")
    
    locacao_opcoes = ["L1", "L2", "L3", "L4", "L5"]
    locacao_default = "L3"
    locacao = st.selectbox("Locação (Duplo T)", locacao_opcoes, index=locacao_opcoes.index(locacao_default))
    
    deflexao = st.slider("Ângulo de deflexão (°)", 0, 150, 30, 1)
    
    altura_poste = st.selectbox("Altura do poste (m)", [14, 16, 18, 20, 22, 24, 26, 28, 30, 32], index=4)
    
    st.markdown("---")
    st.subheader("📏 2. VÃOS E CIRCUITOS")
    
    vao_re = st.number_input("Vão ré (m)", min_value=1, max_value=600, value=100, step=1)
    vao_vante = st.number_input("Vão vante (m)", min_value=1, max_value=600, value=100, step=1)
    num_circuitos = st.selectbox("Circuitos", ["Circuito Simples", "Circuito Duplo"], index=0)
    is_duplo = (num_circuitos == "Circuito Duplo")
    
    st.markdown("---")
    st.subheader("🌡️ 3. PARÂMETROS METEOROLÓGICOS")
    
    temp_eds = st.number_input("Temp EDS (°C)", value=20.0, step=0.5)
    temp_min = st.number_input("Temp mínima (°C)", value=5.0, step=0.5)
    creep_min = st.number_input("Creep temp mín (°C)", value=0.0, step=0.5)
    temp_vento = st.number_input("Temp vento (°C)", value=15.0, step=0.5)
    pressao_vento = st.number_input("Pressão vento (kgf/m²)", value=30.0, step=1.0)
    pressao_vento_reduzida = st.number_input("Pressão vento reduzida (kgf/m²)", value=15.0, step=1.0)
    
    st.markdown("---")
    st.subheader("🔒 4. FATOR DE SEGURANÇA")
    
    fator_cagaco = st.slider("Fator cagaço (%)", 0, 20, 0, 1)

with col2:
    st.subheader("⚡ 5. CABO CONDUTOR")
    
    cabos_nomes = [cabos_data[i]['nome'] for i in cabos_data.keys()]
    cabo_circ1 = st.selectbox("Tipo de cabo (Circ1)", cabos_nomes, index=1)
    caboid1 = [i for i in cabos_data.keys() if cabos_data[i]['nome'] == cabo_circ1][0]
    cr1 = cabos_data[caboid1]['CR']
    
    perc_re_circ1 = st.number_input("%CR ré (Circ1)", min_value=0.0, max_value=100.0, value=5.0, step=0.5)
    perc_vante_circ1 = st.number_input("%CR vante (Circ1)", min_value=0.0, max_value=100.0, value=5.0, step=0.5)
    
    if is_duplo:
        cabo_circ2 = st.selectbox("Tipo de cabo (Circ2)", cabos_nomes, index=1)
        caboid2 = [i for i in cabos_data.keys() if cabos_data[i]['nome'] == cabo_circ2][0]
        cr2 = cabos_data[caboid2]['CR']
        perc_re_circ2 = st.number_input("%CR ré (Circ2)", min_value=0.0, max_value=100.0, value=5.0, step=0.5)
        perc_vante_circ2 = st.number_input("%CR vante (Circ2)", min_value=0.0, max_value=100.0, value=5.0, step=0.5)
    else:
        cr2 = 0
        perc_re_circ2 = 0
        perc_vante_circ2 = 0
    
    st.markdown("---")
    st.subheader("📏 6. GEOMETRIA DAS FASES")
    
    st.markdown("**Circuito 1:**")
    dist_f1 = st.number_input("Fase 1 ao topo (m)", value=0.20, step=0.05)
    dist_f2 = st.number_input("Fase 2 ao topo (m)", value=2.20, step=0.05)
    dist_f3 = st.number_input("Fase 3 ao topo (m)", value=4.20, step=0.05)
    
    if is_duplo:
        st.markdown("**Circuito 2:**")
        dist_f4 = st.number_input("Fase 4 ao topo (m)", value=6.20, step=0.05)
        dist_f5 = st.number_input("Fase 5 ao topo (m)", value=8.20, step=0.05)
        dist_f6 = st.number_input("Fase 6 ao topo (m)", value=10.20, step=0.05)
        dist_fases = [dist_f1, dist_f2, dist_f3, dist_f4, dist_f5, dist_f6]
    else:
        dist_fases = [dist_f1, dist_f2, dist_f3]
    
    st.markdown("---")
    st.subheader("⚡ 7. CABO PARA-RAIOS")
    
    tem_pr = st.checkbox("Possui cabo para-raios?")
    
    if tem_pr:
        tipo_pr = st.selectbox("Tipo de PR", ["Tipo A", "Tipo B", "Tipo C"], index=0)
        cr_pr = cabos_pr_data[tipo_pr]['CR']
        dist_pr = st.number_input("Dist. PR ao topo (m)", value=0.20, step=0.05)
        perc_pr_re = st.number_input("%CR ré (PR)", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
        perc_pr_vante = st.number_input("%CR vante (PR)", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
    else:
        cr_pr = 0
        dist_pr = None
        perc_pr_re = 0
        perc_pr_vante = 0
    
    st.markdown("---")
    st.subheader("⚙️ 8. CONFIGURAÇÕES GERAIS")
    
    cabos_por_fase = st.selectbox("Cabos por fase", [1, 2], index=0)

# ===================================================
# BOTÃO DE CÁLCULO
# ===================================================

if st.button("🔍 Calcular Esforço no Poste", type="primary", use_container_width=True):
    try:
        # ===================================================
        # DADOS DOS CABOS
        # ===================================================
        peso1 = cabos_data[caboid1]['peso']
        E1 = cabos_data[caboid1]['E']
        S1 = cabos_data[caboid1]['area']
        alpha1 = cabos_data[caboid1]['alpha']
        D1 = cabos_data[caboid1]['D']
        
        if is_duplo:
            peso2 = cabos_data[caboid2]['peso']
            E2 = cabos_data[caboid2]['E']
            S2 = cabos_data[caboid2]['area']
            alpha2 = cabos_data[caboid2]['alpha']
            D2 = cabos_data[caboid2]['D']
        else:
            peso2 = E2 = S2 = alpha2 = D2 = 0
        
        # ===================================================
        # TRAÇÕES EDS
        # ===================================================
        T_circ1_re_eds = cr1 * (perc_re_circ1 / 100)
        T_circ1_vante_eds = cr1 * (perc_vante_circ1 / 100) if fixacao == 'ancoragem' else T_circ1_re_eds
        
        if is_duplo:
            T_circ2_re_eds = cr2 * (perc_re_circ2 / 100)
            T_circ2_vante_eds = cr2 * (perc_vante_circ2 / 100) if fixacao == 'ancoragem' else T_circ2_re_eds
        else:
            T_circ2_re_eds = T_circ2_vante_eds = 0
        
        if tem_pr:
            T_pr_re_eds = cr_pr * (perc_pr_re / 100)
            T_pr_vante_eds = cr_pr * (perc_pr_vante / 100) if fixacao == 'ancoragem' else T_pr_re_eds
        else:
            T_pr_re_eds = T_pr_vante_eds = 0
        
        # ===================================================
        # COEFICIENTES K
        # ===================================================
        dist_pr = dist_pr if tem_pr else None
        K_fases, K_pr, X, parte_aerea = calcular_coeficientes_k(dist_fases, dist_pr, altura_poste)
        
        K_circ1 = sum(K_fases[0:3]) * cabos_por_fase
        if is_duplo:
            K_circ2 = sum(K_fases[3:6]) * cabos_por_fase
        else:
            K_circ2 = 0
        
        delta = deflexao if fixacao == 'ancoragem' else 0
        ang_re = 180
        ang_vante = delta
        
        # ===================================================
        # HIPÓTESE 1: EDS
        # ===================================================
        F_re_eds = K_circ1 * T_circ1_re_eds + K_circ2 * T_circ2_re_eds + K_pr * T_pr_re_eds
        F_vante_eds = K_circ1 * T_circ1_vante_eds + K_circ2 * T_circ2_vante_eds + K_pr * T_pr_vante_eds
        
        F_re_eds_vec = forca_complexa(F_re_eds, ang_re)
        F_vante_eds_vec = forca_complexa(F_vante_eds, ang_vante)
        R_eds_vec = F_re_eds_vec + F_vante_eds_vec
        R_eds = abs(R_eds_vec)
        theta_eds = np.degrees(np.angle(R_eds_vec))
        if theta_eds < 0:
            theta_eds += 360
        
        # ===================================================
        # HIPÓTESE 2: TEMPERATURA MÍNIMA
        # ===================================================
        temp_min_eq = temp_min - creep_min
        
        T_circ1_re_min = mudanca_estado(T_circ1_re_eds, temp_eds, temp_min_eq, peso1, peso1, vao_re, E1, S1, alpha1)
        T_circ1_vante_min = mudanca_estado(T_circ1_vante_eds, temp_eds, temp_min_eq, peso1, peso1, vao_vante, E1, S1, alpha1)
        
        if is_duplo:
            T_circ2_re_min = mudanca_estado(T_circ2_re_eds, temp_eds, temp_min_eq, peso2, peso2, vao_re, E2, S2, alpha2)
            T_circ2_vante_min = mudanca_estado(T_circ2_vante_eds, temp_eds, temp_min_eq, peso2, peso2, vao_vante, E2, S2, alpha2)
        else:
            T_circ2_re_min = T_circ2_vante_min = 0
        
        if tem_pr:
            T_pr_re_min = mudanca_estado(T_pr_re_eds, temp_eds, temp_min_eq, cabos_pr_data[tipo_pr]['peso'], cabos_pr_data[tipo_pr]['peso'], vao_re, cabos_pr_data[tipo_pr]['E'], cabos_pr_data[tipo_pr]['area'], cabos_pr_data[tipo_pr]['alpha'])
            T_pr_vante_min = mudanca_estado(T_pr_vante_eds, temp_eds, temp_min_eq, cabos_pr_data[tipo_pr]['peso'], cabos_pr_data[tipo_pr]['peso'], vao_vante, cabos_pr_data[tipo_pr]['E'], cabos_pr_data[tipo_pr]['area'], cabos_pr_data[tipo_pr]['alpha'])
        else:
            T_pr_re_min = T_pr_vante_min = 0
        
        F_re_min = K_circ1 * T_circ1_re_min + K_circ2 * T_circ2_re_min + K_pr * T_pr_re_min
        F_vante_min = K_circ1 * T_circ1_vante_min + K_circ2 * T_circ2_vante_min + K_pr * T_pr_vante_min
        
        F_re_min_vec = forca_complexa(F_re_min, ang_re)
        F_vante_min_vec = forca_complexa(F_vante_min, ang_vante)
        R_min_vec = F_re_min_vec + F_vante_min_vec
        R_min = abs(R_min_vec)
        theta_min = np.degrees(np.angle(R_min_vec))
        if theta_min < 0:
            theta_min += 360
        
        # ===================================================
        # HIPÓTESE 3: VENTO MÁXIMO
        # ===================================================
        S_arrasto = 0
        for i in range(len(K_fases)):
            if i < 3:
                S_arrasto += D1 * K_fases[i] * cabos_por_fase
            else:
                if is_duplo:
                    S_arrasto += D2 * K_fases[i] * cabos_por_fase
        if tem_pr:
            S_arrasto += cabos_pr_data[tipo_pr]['D'] * K_pr
        
        fv_max = D1 * pressao_vento
        peso_composto = np.sqrt(peso1**2 + fv_max**2)
        
        T_circ1_re_vento = mudanca_estado(T_circ1_re_eds, temp_eds, temp_vento, peso1, peso_composto, vao_re, E1, S1, alpha1)
        T_circ1_vante_vento = mudanca_estado(T_circ1_vante_eds, temp_eds, temp_vento, peso1, peso_composto, vao_vante, E1, S1, alpha1)
        
        if is_duplo:
            fv_max2 = D2 * pressao_vento
            peso_composto2 = np.sqrt(peso2**2 + fv_max2**2)
            T_circ2_re_vento = mudanca_estado(T_circ2_re_eds, temp_eds, temp_vento, peso2, peso_composto2, vao_re, E2, S2, alpha2)
            T_circ2_vante_vento = mudanca_estado(T_circ2_vante_eds, temp_eds, temp_vento, peso2, peso_composto2, vao_vante, E2, S2, alpha2)
        else:
            T_circ2_re_vento = T_circ2_vante_vento = 0
        
        if tem_pr:
            fv_max_pr = cabos_pr_data[tipo_pr]['D'] * pressao_vento
            peso_composto_pr = np.sqrt(cabos_pr_data[tipo_pr]['peso']**2 + fv_max_pr**2)
            T_pr_re_vento = mudanca_estado(T_pr_re_eds, temp_eds, temp_vento, cabos_pr_data[tipo_pr]['peso'], peso_composto_pr, vao_re, cabos_pr_data[tipo_pr]['E'], cabos_pr_data[tipo_pr]['area'], cabos_pr_data[tipo_pr]['alpha'])
            T_pr_vante_vento = mudanca_estado(T_pr_vante_eds, temp_eds, temp_vento, cabos_pr_data[tipo_pr]['peso'], peso_composto_pr, vao_vante, cabos_pr_data[tipo_pr]['E'], cabos_pr_data[tipo_pr]['area'], cabos_pr_data[tipo_pr]['alpha'])
        else:
            T_pr_re_vento = T_pr_vante_vento = 0
        
        F_re_vento_cabos = K_circ1 * T_circ1_re_vento + K_circ2 * T_circ2_re_vento + K_pr * T_pr_re_vento
        F_vante_vento_cabos = K_circ1 * T_circ1_vante_vento + K_circ2 * T_circ2_vante_vento + K_pr * T_pr_vante_vento
        
        ang_vento_critico, _ = encontrar_angulo_critico_vento(
            F_re_vento_cabos, F_vante_vento_cabos, ang_re, ang_vante,
            S_arrasto, pressao_vento, vao_re, vao_vante)
        
        seno_re = abs(np.sin(np.radians(ang_vento_critico - ang_re)))
        seno_vante = abs(np.sin(np.radians(ang_vento_critico - ang_vante)))
        F_vento_mag = (pressao_vento/2) * (vao_re * seno_re + vao_vante * seno_vante) * S_arrasto
        
        F_re_vento = F_re_vento_cabos + F_vento_mag * np.cos(np.radians(ang_vento_critico))
        F_vante_vento = F_vante_vento_cabos + F_vento_mag * np.sin(np.radians(ang_vento_critico))
        
        F_re_vento_vec = forca_complexa(F_re_vento_cabos, ang_re)
        F_vante_vento_vec = forca_complexa(F_vante_vento_cabos, ang_vante)
        F_vento_vec = forca_complexa(F_vento_mag, ang_vento_critico)
        R_vento_vec = F_re_vento_vec + F_vante_vento_vec + F_vento_vec
        R_vento = abs(R_vento_vec)
        theta_vento = np.degrees(np.angle(R_vento_vec))
        if theta_vento < 0:
            theta_vento += 360
        
        # ===================================================
        # HIPÓTESE 4: CABO CONDUTOR ROMPIDO
        # ===================================================
        num_fases = len(K_fases)
        indice_fase_mais_alta = np.argmax(K_fases)
        
        fv_max_red = D1 * pressao_vento_reduzida
        peso_composto_red = np.sqrt(peso1**2 + fv_max_red**2)
        
        T_circ1_re_rompido = mudanca_estado(T_circ1_re_eds, temp_eds, temp_vento, peso1, peso_composto_red, vao_re, E1, S1, alpha1)
        T_circ1_vante_rompido = mudanca_estado(T_circ1_vante_eds, temp_eds, temp_vento, peso1, peso_composto_red, vao_vante, E1, S1, alpha1)
        
        if is_duplo:
            fv_max_red2 = D2 * pressao_vento_reduzida
            peso_composto_red2 = np.sqrt(peso2**2 + fv_max_red2**2)
            T_circ2_re_rompido = mudanca_estado(T_circ2_re_eds, temp_eds, temp_vento, peso2, peso_composto_red2, vao_re, E2, S2, alpha2)
            T_circ2_vante_rompido = mudanca_estado(T_circ2_vante_eds, temp_eds, temp_vento, peso2, peso_composto_red2, vao_vante, E2, S2, alpha2)
        else:
            T_circ2_re_rompido = T_circ2_vante_rompido = 0
        
        if tem_pr:
            fv_max_pr_red = cabos_pr_data[tipo_pr]['D'] * pressao_vento_reduzida
            peso_composto_pr_red = np.sqrt(cabos_pr_data[tipo_pr]['peso']**2 + fv_max_pr_red**2)
            T_pr_re_rompido = mudanca_estado(T_pr_re_eds, temp_eds, temp_vento, cabos_pr_data[tipo_pr]['peso'], peso_composto_pr_red, vao_re, cabos_pr_data[tipo_pr]['E'], cabos_pr_data[tipo_pr]['area'], cabos_pr_data[tipo_pr]['alpha'])
            T_pr_vante_rompido = mudanca_estado(T_pr_vante_eds, temp_eds, temp_vento, cabos_pr_data[tipo_pr]['peso'], peso_composto_pr_red, vao_vante, cabos_pr_data[tipo_pr]['E'], cabos_pr_data[tipo_pr]['area'], cabos_pr_data[tipo_pr]['alpha'])
        else:
            T_pr_re_rompido = T_pr_vante_rompido = 0
        
        # Cenário A: rompe no ré
        T_re_rompido_A = []
        for i in range(num_fases):
            if i == indice_fase_mais_alta:
                T_re_rompido_A.append(0)
            else:
                T_re_rompido_A.append(T_circ1_re_rompido if i < 3 else T_circ2_re_rompido)
        
        T_vante_rompido_A = [T_circ1_vante_rompido if i < 3 else T_circ2_vante_rompido for i in range(num_fases)]
        
        F_re_rompido_A = sum([K_fases[i] * T_re_rompido_A[i] for i in range(num_fases)]) + (K_pr * T_pr_re_rompido if tem_pr else 0)
        F_vante_rompido_A = sum([K_fases[i] * T_vante_rompido_A[i] for i in range(num_fases)]) + (K_pr * T_pr_vante_rompido if tem_pr else 0)
        
        # Cenário B: rompe no vante
        T_re_rompido_B = [T_circ1_re_rompido if i < 3 else T_circ2_re_rompido for i in range(num_fases)]
        
        T_vante_rompido_B = []
        for i in range(num_fases):
            if i == indice_fase_mais_alta:
                T_vante_rompido_B.append(0)
            else:
                T_vante_rompido_B.append(T_circ1_vante_rompido if i < 3 else T_circ2_vante_rompido)
        
        F_re_rompido_B = sum([K_fases[i] * T_re_rompido_B[i] for i in range(num_fases)]) + (K_pr * T_pr_re_rompido if tem_pr else 0)
        F_vante_rompido_B = sum([K_fases[i] * T_vante_rompido_B[i] for i in range(num_fases)]) + (K_pr * T_pr_vante_rompido if tem_pr else 0)
        
        R_rompido_A = abs(forca_complexa(F_re_rompido_A, ang_re) + forca_complexa(F_vante_rompido_A, ang_vante))
        R_rompido_B = abs(forca_complexa(F_re_rompido_B, ang_re) + forca_complexa(F_vante_rompido_B, ang_vante))
        
        if R_rompido_A >= R_rompido_B:
            pior_cenario = "Rompimento no ré"
            F_re_rompido_final = F_re_rompido_A
            F_vante_rompido_final = F_vante_rompido_A
        else:
            pior_cenario = "Rompimento no vante"
            F_re_rompido_final = F_re_rompido_B
            F_vante_rompido_final = F_vante_rompido_B
        
        # S_arrasto reduzido
        S_arrasto_reduzido = 0
        for i in range(num_fases):
            if i == indice_fase_mais_alta:
                continue
            else:
                if i < 3:
                    S_arrasto_reduzido += D1 * K_fases[i] * cabos_por_fase
                else:
                    if is_duplo:
                        S_arrasto_reduzido += D2 * K_fases[i] * cabos_por_fase
        if tem_pr:
            S_arrasto_reduzido += cabos_pr_data[tipo_pr]['D'] * K_pr
        
        ang_vento_rompido_critico, _ = encontrar_angulo_critico_vento(
            F_re_rompido_final, F_vante_rompido_final, ang_re, ang_vante,
            S_arrasto_reduzido, pressao_vento_reduzida, vao_re, vao_vante)
        
        seno_re_rompido = abs(np.sin(np.radians(ang_vento_rompido_critico - ang_re)))
        seno_vante_rompido = abs(np.sin(np.radians(ang_vento_rompido_critico - ang_vante)))
        F_vento_rompido_mag = (pressao_vento_reduzida/2) * (vao_re * seno_re_rompido + vao_vante * seno_vante_rompido) * S_arrasto_reduzido
        
        F_re_rompido_total = F_re_rompido_final + F_vento_rompido_mag * np.cos(np.radians(ang_vento_rompido_critico))
        F_vante_rompido_total = F_vante_rompido_final + F_vento_rompido_mag * np.sin(np.radians(ang_vento_rompido_critico))
        
        F_re_rompido_vec = forca_complexa(F_re_rompido_final, ang_re)
        F_vante_rompido_vec = forca_complexa(F_vante_rompido_final, ang_vante)
        F_vento_rompido_vec = forca_complexa(F_vento_rompido_mag, ang_vento_rompido_critico)
        R_rompido_vec = F_re_rompido_vec + F_vante_rompido_vec + F_vento_rompido_vec
        R_rompido = abs(R_rompido_vec)
        theta_rompido = np.degrees(np.angle(R_rompido_vec))
        if theta_rompido < 0:
            theta_rompido += 360
        
        # ===================================================
        # CORREÇÃO DUPLO T
        # ===================================================
        comerciais = [1500, 2000, 2500, 3000, 4000]
        
        if tipo_poste == 'Duplo T':
            bissetriz = (180 + delta) / 2
            if locacao in ['L1', 'L3']:
                theta_face = bissetriz
            elif locacao in ['L2', 'L4']:
                theta_face = bissetriz - 90
            else:
                theta_face = ang_re if F_re_eds >= F_vante_eds else ang_vante
            
            fator_eds = fator_correcao_duplo_t(theta_eds, theta_face)
            R_eds_corrigido = R_eds / fator_eds
            
            fator_min = fator_correcao_duplo_t(theta_min, theta_face)
            R_min_corrigido = R_min / fator_min
            
            fator_vento = fator_correcao_duplo_t(theta_vento, theta_face)
            R_vento_corrigido = R_vento / fator_vento
            
            fator_rompido = fator_correcao_duplo_t(theta_rompido, theta_face)
            R_rompido_corrigido = R_rompido / fator_rompido
        else:
            theta_face = None
            fator_eds = fator_min = fator_vento = fator_rompido = 1.0
            R_eds_corrigido = R_eds
            R_min_corrigido = R_min
            R_vento_corrigido = R_vento
            R_rompido_corrigido = R_rompido
        
        # FATOR CAGAÇO
        fator_cagaco_float = fator_cagaco / 100
        R_eds_corrigido *= (1 + fator_cagaco_float)
        R_min_corrigido *= (1 + fator_cagaco_float)
        R_vento_corrigido *= (1 + fator_cagaco_float)
        R_rompido_corrigido *= (1 + fator_cagaco_float)
        
        # CONVERSÃO kgf → daN e SOBRECARGA
        R_eds_daN = R_eds_corrigido * KGf_TO_DAN
        R_min_daN = R_min_corrigido * KGf_TO_DAN / (1 + SOBRECARGA_TRANSITORIA)
        R_vento_daN = R_vento_corrigido * KGf_TO_DAN / (1 + SOBRECARGA_TRANSITORIA)
        R_rompido_daN = R_rompido_corrigido * KGf_TO_DAN / (1 + SOBRECARGA_TRANSITORIA)
        
        # POSTE RECOMENDADO
        poste_eds = min([p for p in comerciais if p >= R_eds_daN]) if R_eds_daN <= 4000 else 4000
        poste_min = min([p for p in comerciais if p >= R_min_daN]) if R_min_daN <= 4000 else 4000
        poste_vento = min([p for p in comerciais if p >= R_vento_daN]) if R_vento_daN <= 4000 else 4000
        poste_rompido = min([p for p in comerciais if p >= R_rompido_daN]) if R_rompido_daN <= 4000 else 4000
        poste_recomendado = max(poste_eds, poste_min, poste_vento, poste_rompido)
        
        # ===================================================
        # EXIBIR RESULTADOS
        # ===================================================
        st.markdown("---")
        st.subheader("📊 RESULTADOS")
        
        # Tabela
        dados_tabela = [
            {'Hipótese': 'EDS', 'Esforço (daN)': f"{R_eds_daN:.1f}", 'Poste': f"{poste_eds} daN"},
            {'Hipótese': 'TEMP MÍNIMA', 'Esforço (daN)': f"{R_min_daN:.1f}", 'Poste': f"{poste_min} daN"},
            {'Hipótese': 'VENTO MÁXIMO', 'Esforço (daN)': f"{R_vento_daN:.1f}", 'Poste': f"{poste_vento} daN"},
            {'Hipótese': 'CABO ROMPIDO', 'Esforço (daN)': f"{R_rompido_daN:.1f}", 'Poste': f"{poste_rompido} daN"},
        ]
        
        df = pd.DataFrame(dados_tabela)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Conclusão
        st.markdown("---")
        st.subheader("🎯 CONCLUSÃO")
        
        st.metric("Poste comercial recomendado", f"{poste_recomendado} daN")
        
        # Alerta
        if R_eds_daN > 4000 or R_min_daN > 4000 or R_vento_daN > 4000 or R_rompido_daN > 4000:
            st.warning("⚠️ ATENÇÃO: Esforço calculado > 4000 daN. Não há poste comercial disponível com esta capacidade.")
        
        # Diagramas em abas
        st.markdown("---")
        st.subheader("📈 DIAGRAMAS VETORIAIS")
        
        tab1, tab2, tab3, tab4 = st.tabs(["EDS", "TEMP MÍNIMA", "VENTO MÁXIMO", "CABO ROMPIDO"])
        
        with tab1:
            fig1 = plotar_diagrama(
                F_re_eds, F_vante_eds, 0, R_eds,
                ang_re, ang_vante, 0, theta_eds,
                tipo_poste, f"EDS - Resultante: {R_eds_daN:.1f} daN",
                theta_face if tipo_poste == 'Duplo T' else None,
                fator_eds if tipo_poste == 'Duplo T' else None
            )
            st.pyplot(fig1)
            plt.close(fig1)
        
        with tab2:
            fig2 = plotar_diagrama(
                F_re_min, F_vante_min, 0, R_min,
                ang_re, ang_vante, 0, theta_min,
                tipo_poste, f"TEMP MÍNIMA - Resultante: {R_min_daN:.1f} daN",
                theta_face if tipo_poste == 'Duplo T' else None,
                fator_min if tipo_poste == 'Duplo T' else None
            )
            st.pyplot(fig2)
            plt.close(fig2)
        
        with tab3:
            fig3 = plotar_diagrama(
                F_re_vento_cabos, F_vante_vento_cabos, F_vento_mag, R_vento,
                ang_re, ang_vante, ang_vento_critico, theta_vento,
                tipo_poste, f"VENTO MÁXIMO - Resultante: {R_vento_daN:.1f} daN",
                theta_face if tipo_poste == 'Duplo T' else None,
                fator_vento if tipo_poste == 'Duplo T' else None
            )
            st.pyplot(fig3)
            plt.close(fig3)
        
        with tab4:
            fig4 = plotar_diagrama(
                F_re_rompido_final, F_vante_rompido_final, F_vento_rompido_mag, R_rompido,
                ang_re, ang_vante, ang_vento_rompido_critico, theta_rompido,
                tipo_poste, f"CABO ROMPIDO - Resultante: {R_rompido_daN:.1f} daN",
                theta_face if tipo_poste == 'Duplo T' else None,
                fator_rompido if tipo_poste == 'Duplo T' else None
            )
            st.pyplot(fig4)
            plt.close(fig4)
        
    except Exception as e:
        st.error(f"❌ Erro no cálculo: {e}")
        st.exception(e)
