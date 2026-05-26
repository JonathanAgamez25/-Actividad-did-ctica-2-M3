"""
╔══════════════════════════════════════════════════════════════════╗
║      SIMULACIÓN DE COLAS M/M/1 — BANCO DE COLOMBIA              ║
║      3 cajeros independientes | 8 horas/día | 10 réplicas        ║
╚══════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ══════════════════════════════════════════════════════════════════
#  PARÁMETROS GLOBALES
# ══════════════════════════════════════════════════════════════════
TIEMPO_OPERACION = 480      # minutos (8 horas)
N_REPLICAS       = 10
UMBRAL_ESPERA    = 5.0      # minutos — referencia para punto 4
CSV_SALIDA       = Path(__file__).parent / "resultados_simulacion.csv"

# Tabla 1 — (media_servicio μ, media_llegada λ, probabilidad)
TIPOS_RETIRO = {
    "Rápido":    {"mu": 1, "lam": 1, "prob": 0.23},
    "Normal":    {"mu": 2, "lam": 2, "prob": 0.40},
    "Lento":     {"mu": 3, "lam": 3, "prob": 0.17},
    "Muy lento": {"mu": 4, "lam": 3, "prob": 0.20},
}
TIPOS_PAGO = {
    "Rápido":    {"mu": 3, "lam": 1, "prob": 0.10},
    "Normal":    {"mu": 3, "lam": 2, "prob": 0.20},
    "Lento":     {"mu": 5, "lam": 3, "prob": 0.30},
    "Muy lento": {"mu": 7, "lam": 4, "prob": 0.40},
}

# ══════════════════════════════════════════════════════════════════
#  ENTIDADES
# ══════════════════════════════════════════════════════════════════
@dataclass
class Cliente:
    tipo_accion:   str
    tipo_usuario:  str
    t_llegada:     float
    t_servicio:    float
    t_inicio:      float = 0.0
    t_salida:      float = 0.0

    @property
    def espera(self):   return self.t_inicio - self.t_llegada
    @property
    def en_sistema(self): return self.t_salida - self.t_llegada


@dataclass
class ResultadoCajero:
    cajero_id:      int
    tipo_atencion:  str
    clientes:       List[Cliente] = field(default_factory=list)

    # ── métricas ──────────────────────────────────────────────────
    @property
    def n(self):           return len(self.clientes)
    @property
    def ws(self):          return np.mean([c.t_servicio for c in self.clientes]) if self.clientes else 0
    @property
    def wq(self):          return np.mean([c.espera     for c in self.clientes]) if self.clientes else 0
    @property
    def w(self):           return np.mean([c.en_sistema for c in self.clientes]) if self.clientes else 0
    @property
    def utilizacion(self):
        return sum(c.t_servicio for c in self.clientes) / TIEMPO_OPERACION if self.clientes else 0
    @property
    def n_retiro(self):    return sum(1 for c in self.clientes if c.tipo_accion == "retiro")
    @property
    def n_pago(self):      return sum(1 for c in self.clientes if c.tipo_accion == "pago")


# ══════════════════════════════════════════════════════════════════
#  GENERADORES
# ══════════════════════════════════════════════════════════════════
def _seleccionar_tipo(accion: str, rng) -> tuple:
    tabla = TIPOS_RETIRO if accion == "retiro" else TIPOS_PAGO
    tipos = list(tabla.keys())
    probs = [tabla[t]["prob"] for t in tipos]
    tipo  = rng.choice(tipos, p=probs)
    t_srv = rng.exponential(tabla[tipo]["mu"])
    t_ent = rng.exponential(tabla[tipo]["lam"])
    return tipo, t_srv, t_ent


# ══════════════════════════════════════════════════════════════════
#  SIMULACIÓN DE UN CAJERO M/M/1
# ══════════════════════════════════════════════════════════════════
def simular_cajero(cajero_id: int, tipo_atencion: str, rng) -> ResultadoCajero:
    resultado   = ResultadoCajero(cajero_id=cajero_id, tipo_atencion=tipo_atencion)
    cajero_libre = 0.0
    t_actual     = 0.0

    while t_actual < TIEMPO_OPERACION:
        tipo, t_srv, t_entre = _seleccionar_tipo(tipo_atencion, rng)
        t_actual += t_entre
        if t_actual >= TIEMPO_OPERACION:
            break

        c          = Cliente(tipo_atencion, tipo, t_actual, t_srv)
        c.t_inicio = max(t_actual, cajero_libre)
        c.t_salida = c.t_inicio + t_srv

        if c.t_inicio < TIEMPO_OPERACION:
            cajero_libre = c.t_salida
            resultado.clientes.append(c)

    return resultado


# ══════════════════════════════════════════════════════════════════
#  CONFIGURACIONES
# ══════════════════════════════════════════════════════════════════
def _correr_configuracion(tipos: list, seed: int) -> List[ResultadoCajero]:
    rng = np.random.default_rng(seed)
    return [simular_cajero(i + 1, t, rng) for i, t in enumerate(tipos)]


def ejecutar_simulacion():
    semillas = [100 * (i + 1) for i in range(N_REPLICAS)]
    res_A = [_correr_configuracion(["retiro", "pago", "pago"], s) for s in semillas]
    res_B = [_correr_configuracion(["retiro", "retiro", "pago"], s) for s in semillas]
    return res_A, res_B


# ══════════════════════════════════════════════════════════════════
#  HELPERS DE PRESENTACIÓN
# ══════════════════════════════════════════════════════════════════
SEP_DOBLE  = "═" * 65
SEP_SIMPLE = "─" * 65
SEP_PUNTO  = "·" * 65

def _titulo(texto: str) -> None:
    print(f"\n{SEP_DOBLE}")
    print(f"  {texto}")
    print(SEP_DOBLE)

def _subtitulo(texto: str) -> None:
    print(f"\n  ┌─ {texto} {'─' * (55 - len(texto))}┐")

def _fila(etiqueta: str, valor: str, alerta: str = "") -> None:
    print(f"  │  {etiqueta:<35} {valor:<15} {alerta}")

def _cierre() -> None:
    print(f"  └{SEP_SIMPLE[1:56]}┘")

def _resumen_cajero(cajero_id, tipo, ws, wq, util, alerta="") -> None:
    print(f"  │  Cajero {cajero_id} [{tipo:<6}]   "
          f"Ws={ws:>6.3f} min   Wq={wq:>7.3f} min   ρ={util:>6.2%}  {alerta}")


# ══════════════════════════════════════════════════════════════════
#  ESTADÍSTICAS POR CAJERO (para reutilizar)
# ══════════════════════════════════════════════════════════════════
def _promedios_cajero(resultados, cajero_idx):
    ws_vals   = [rep[cajero_idx].ws          for rep in resultados]
    wq_vals   = [rep[cajero_idx].wq          for rep in resultados]
    util_vals = [rep[cajero_idx].utilizacion for rep in resultados]
    tipo      = resultados[0][cajero_idx].tipo_atencion
    return tipo, np.mean(ws_vals), np.mean(wq_vals), np.mean(util_vals)


# ══════════════════════════════════════════════════════════════════
#  PUNTO 1 — Tiempo promedio de atención
# ══════════════════════════════════════════════════════════════════
def punto1(res_A, res_B):
    _titulo("PUNTO 1 │ Tiempo promedio de atención por cajero")
    for resultados, nombre in [(res_A, "Configuración A  (1 Retiro — 2 Pagos)"),
                                (res_B, "Configuración B  (2 Retiros — 1 Pago)")]:
        _subtitulo(nombre)
        ws_por_cajero = {}
        for idx in range(3):
            tipo, ws, wq, util = _promedios_cajero(resultados, idx)
            ws_por_cajero[idx + 1] = (tipo, ws)
            _fila(f"Cajero {idx+1} [{tipo}]", f"{ws:.4f} min")
        print(f"  │")
        min_id = min(ws_por_cajero, key=lambda k: ws_por_cajero[k][1])
        max_id = max(ws_por_cajero, key=lambda k: ws_por_cajero[k][1])
        _fila("  ↓ MENOR tiempo de atención",
              f"Cajero {min_id} → {ws_por_cajero[min_id][1]:.4f} min")
        _fila("  ↑ MAYOR tiempo de atención",
              f"Cajero {max_id} → {ws_por_cajero[max_id][1]:.4f} min")
        _cierre()


# ══════════════════════════════════════════════════════════════════
#  PUNTO 2 — Promedio de usuarios por tipo
# ══════════════════════════════════════════════════════════════════
def punto2(res_A, res_B):
    _titulo("PUNTO 2 │ Promedio de usuarios por tipo (todos los cajeros)")
    for resultados, nombre in [(res_A, "Configuración A  (1 Retiro — 2 Pagos)"),
                                (res_B, "Configuración B  (2 Retiros — 1 Pago)")]:
        _subtitulo(nombre)
        retiros = [sum(rep[i].n_retiro for i in range(3)) for rep in resultados]
        pagos   = [sum(rep[i].n_pago   for i in range(3)) for rep in resultados]
        _fila("Promedio retiros / día",
              f"{np.mean(retiros):.2f}", f"(σ = {np.std(retiros):.2f})")
        _fila("Promedio pagos   / día",
              f"{np.mean(pagos):.2f}",   f"(σ = {np.std(pagos):.2f})")
        _fila("Promedio total   / día",
              f"{np.mean(retiros)+np.mean(pagos):.2f}", "")
        _cierre()


# ══════════════════════════════════════════════════════════════════
#  PUNTO 3 — Totales por réplica
# ══════════════════════════════════════════════════════════════════
def punto3(res_A, res_B):
    _titulo("PUNTO 3 │ Total de usuarios por tipo en cada réplica")
    for resultados, nombre in [(res_A, "Configuración A  (1 Retiro — 2 Pagos)"),
                                (res_B, "Configuración B  (2 Retiros — 1 Pago)")]:
        _subtitulo(nombre)
        print(f"  │  {'Réplica':>7}  {'Retiros':>8}  {'Pagos':>8}  {'Total':>8}")
        print(f"  │  {'─'*7}  {'─'*8}  {'─'*8}  {'─'*8}")
        min_total, min_rep_data = float("inf"), None
        for i, rep in enumerate(resultados):
            r = sum(rep[j].n_retiro for j in range(3))
            pg = sum(rep[j].n_pago  for j in range(3))
            tot = r + pg
            marca = "  ← mínimo" if tot < min_total else ""
            if tot < min_total:
                min_total, min_rep_data = tot, (i+1, r, pg, tot)
            print(f"  │  {i+1:>7}  {r:>8}  {pg:>8}  {tot:>8}{marca}")
        print(f"  │")
        _fila("  ★ Réplica con menos usuarios",
              f"Réplica {min_rep_data[0]}",
              f"(R={min_rep_data[1]}, P={min_rep_data[2]}, Total={min_rep_data[3]})")
        _cierre()


# ══════════════════════════════════════════════════════════════════
#  PUNTO 4 — ¿Nuevo cajero?
# ══════════════════════════════════════════════════════════════════
def punto4(res_A, res_B):
    _titulo("PUNTO 4 │ ¿Es necesario un cajero adicional?")
    print(f"  Umbral de espera aceptable: {UMBRAL_ESPERA:.0f} min   |   Umbral ρ: 90 %\n")
    for resultados, nombre in [(res_A, "Configuración A  (1 Retiro — 2 Pagos)"),
                                (res_B, "Configuración B  (2 Retiros — 1 Pago)")]:
        _subtitulo(nombre)
        print(f"  │  {'Cajero':<20} {'Wq prom (min)':>15} {'ρ prom':>10}  Estado")
        print(f"  │  {'─'*20} {'─'*15} {'─'*10}  {'─'*12}")
        necesita = False
        for idx in range(3):
            tipo, ws, wq, util = _promedios_cajero(resultados, idx)
            alerta = "⚠ SATURADO" if wq > UMBRAL_ESPERA or util > 0.90 else "OK"
            if alerta.startswith("⚠"):
                necesita = True
            print(f"  │  Cajero {idx+1} [{tipo:<6}]  {wq:>14.3f}  {util:>9.2%}  {alerta}")
        print(f"  │")
        decision = "Sí → Se RECOMIENDA agregar cajero(s) adicional(es)." if necesita \
                   else "No → El sistema opera dentro de parámetros aceptables."
        _fila("  Decisión", decision)
        _cierre()


# ══════════════════════════════════════════════════════════════════
#  PUNTO 5 — Distribución óptima
# ══════════════════════════════════════════════════════════════════
def punto5(res_A, res_B):
    _titulo("PUNTO 5 │ Distribución óptima de cajeros")
    datos = {}
    for resultados, clave, nombre in [
        (res_A, "A", "Configuración A  (1 Retiro — 2 Pagos)"),
        (res_B, "B", "Configuración B  (2 Retiros — 1 Pago)")
    ]:
        _subtitulo(nombre)
        wq_tipo = {"retiro": [], "pago": []}
        ut_tipo = {"retiro": [], "pago": []}
        for idx in range(3):
            tipo, ws, wq, util = _promedios_cajero(resultados, idx)
            wq_tipo[tipo].append(wq)
            ut_tipo[tipo].append(util)
        for tipo in ["retiro", "pago"]:
            if wq_tipo[tipo]:
                wq_m = np.mean(wq_tipo[tipo])
                ut_m = np.mean(ut_tipo[tipo])
                _fila(f"  {tipo.capitalize():>10} — Wq prom", f"{wq_m:.4f} min",
                      f"ρ prom = {ut_m:.2%}")
                datos[f"{clave}_{tipo}_wq"] = wq_m
        _cierre()

    print(f"\n  {SEP_PUNTO}")
    print(f"  ANÁLISIS COMPARATIVO")
    print(f"  {SEP_PUNTO}")
    mejor_retiro = "A" if datos["A_retiro_wq"] <= datos["B_retiro_wq"] else "B"
    mejor_pago   = "A" if datos["A_pago_wq"]   <= datos["B_pago_wq"]   else "B"
    print(f"  Menor espera retiro → Config {mejor_retiro}  "
          f"(A={datos['A_retiro_wq']:.3f} min  vs  B={datos['B_retiro_wq']:.3f} min)")
    print(f"  Menor espera pago   → Config {mejor_pago}   "
          f"(A={datos['A_pago_wq']:.3f} min  vs  B={datos['B_pago_wq']:.3f} min)")
    print(f"\n  ★ RECOMENDACIÓN FINAL:")
    print(f"    Adoptar Configuración A (1 retiro – 2 pagos).")
    print(f"    Los pagos generan el cuello de botella; necesitan más cajeros.")
    print(f"    Se sugiere agregar un 4.° cajero exclusivo para pagos.")
    print(f"  {SEP_PUNTO}\n")


# ══════════════════════════════════════════════════════════════════
#  EXPORTAR CSV
# ══════════════════════════════════════════════════════════════════
def exportar_csv(res_A, res_B):
    filas = []
    for resultados, config in [(res_A, "A (1R-2P)"), (res_B, "B (2R-1P)")]:
        for rep_idx, rep in enumerate(resultados):
            for cajero in rep:
                filas.append({
                    "Configuracion":          config,
                    "Replica":                rep_idx + 1,
                    "Cajero":                 cajero.cajero_id,
                    "Tipo":                   cajero.tipo_atencion,
                    "N_Clientes":             cajero.n,
                    "N_Retiro":               cajero.n_retiro,
                    "N_Pago":                 cajero.n_pago,
                    "Ws_prom_min":            round(cajero.ws, 4),
                    "Wq_prom_min":            round(cajero.wq, 4),
                    "W_prom_min":             round(cajero.w, 4),
                    "Utilizacion_pct":        round(cajero.utilizacion * 100, 2),
                })
    df = pd.DataFrame(filas)
    df.to_csv(CSV_SALIDA, index=False, encoding="utf-8-sig")
    return df


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "█" * 65)
    print("█  SIMULACIÓN M/M/1 — BANCO DE COLOMBIA" + " " * 24 + "█")
    print(f"█  Réplicas: {N_REPLICAS}   |   Horizonte: {TIEMPO_OPERACION} min/día" + " " * 27 + "█")
    print("█" * 65)

    res_A, res_B = ejecutar_simulacion()

    punto1(res_A, res_B)
    punto2(res_A, res_B)
    punto3(res_A, res_B)
    punto4(res_A, res_B)
    punto5(res_A, res_B)

    df = exportar_csv(res_A, res_B)
    print(f"  ✔ CSV exportado en: {CSV_SALIDA}")
    print(f"  ✔ Filas generadas: {len(df)}\n")
