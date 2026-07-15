#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import json
import re
import sqlite3

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, optimize

import pandas as pd


sigma_dir = os.getcwd()
outputs_dir = os.path.abspath("../pyoutputs")


def get_params() -> dict:
    """
    Get the parameters for the simulation.
    """
    pwd = os.getcwd()
    json_path = os.path.join(pwd, "../params.json")

    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            params = json.load(f)
        return params
    else:
        raise FileNotFoundError(f"params.json not found in {json_path}")


def simulate(rocky_filename=None):
    global study
    global project

    params = get_params()
    if not rocky_filename:
        rocky_filename = os.path.join(sigma_dir, glob.glob("*.rocky")[0])

    project = app.OpenProject(rocky_filename)
    study = project.GetStudy()

    solver = study.GetSolver()

    if params.get("processor") == "CPU":
        N_CPUS = int(os.environ["SLURM_CPUS_PER_TASK"])
        solver.SetNumberOfProcessors(N_CPUS)

    if (not study.HasResults()) or (study.CanResumeSimulation()):
        study.StartSimulation(skip_summary=True)
        while study.IsSimulating():
            study.RefreshResults()
            print(f"Simulation Progress: {study.GetProgress():.2f} %", flush=True)
        project.SaveProject()


def postprocess(
    plot=True,
    window_size: int = 5,
) -> float:
    params = get_params()

    box_len = params["particle_box_len"]
    t_shear = params["t_shear"]
    shear_vel = params["shear_vel"]

    # Get the bottom wall
    geometry_collection = study.GetGeometryCollection()
    bottom_wall = geometry_collection.GetGeometry("Compression Wall 2")

    time_arr, power_lst = bottom_wall.GetNumpyCurve("Power")
    power_arr = np.array(power_lst)
    shear_arr = power_arr / (box_len**2 * shear_vel)

    shear_mask = np.where(time_arr >= t_shear)[0]
    shear_arr_masked = shear_arr[shear_mask]

    smoothed_shear_arr = signal.savgol_filter(shear_arr_masked, window_size, 1)
    # time_arr_masked = time_arr[shear_mask]

    # Compute the shear stress
    shear_peaks_idx = signal.find_peaks(shear_arr)[0]
    shear_peaks = shear_arr[shear_peaks_idx]

    if len(shear_peaks) >= 5:
        tau_avg = shear_peaks[-5:].mean().item()
    elif len(shear_peaks) > 0:
        tau_avg = shear_peaks.mean().item()
    else:
        tau_avg = smoothed_shear_arr.mean().item()

    if plot:
        plt.plot(time_arr, shear_arr, label="Shear Stress")
        plt.plot(
            time_arr[shear_mask], smoothed_shear_arr, label="Smoothed Shear Stress"
        )
        plt.axhline(y=tau_avg, color="r", linestyle="--", label="Average Shear Stress")
        if len(shear_peaks) > 0:
            plt.plot(time_arr[shear_peaks_idx], shear_peaks, "x", label="Shear Peaks")
        plt.xlabel("Time (s)")
        plt.ylabel("Shear Stress (Pa)")

        plot_suffix = os.path.basename(sigma_dir).split("_")[-1]

        plt.savefig(
            os.path.join(outputs_dir, f"shear_stress_preshear_{plot_suffix}.png")
        )
        plt.close()

    tau_arr = np.load(os.path.join(outputs_dir, "shear_stresses.npy"))
    sigma_arr = np.load(os.path.join(outputs_dir, "sigma.npy"))

    # Extract sigma from the directory name
    sigma_dirname = os.path.basename(sigma_dir)
    sigma = float(re.search(r"sigma_([0-9]+\.[0-9]+)kpa", sigma_dirname).group(1)) * 1e3

    sigma_idx = np.where(sigma_arr == sigma)[0][0]
    tau_arr[sigma_idx] = tau_avg

    np.save(os.path.join(outputs_dir, "shear_stresses.npy"), tau_arr)

    result_path = os.path.join(sigma_dir, "result.json")
    temp_path = result_path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "sigma": float(sigma_arr[sigma_idx]),
                "sigma_idx": int(sigma_idx),
                "tau": float(tau_avg),
            },
            f,
        )
    os.replace(temp_path, result_path)

    return tau_avg


def _regression_line(sigma: np.ndarray, tau: np.ndarray):
    """
    Fit a linear regression line to the shear stress data.
    Returns the slope and intercept of the line.
    """

    tau_locus = tau[1:]
    sigma_locus = sigma[1:]
    m, c = np.polyfit(sigma_locus, tau_locus, 1, full=False)

    if c <= 0:
        raise ValueError("The y-intercept (c) must be positive for the Mohr circle.")

    # Unconfined Mohr circle
    r_unc = c / ((m**2 + 1) ** 0.5 - m)

    # Confined Mohr circle
    x1, y1 = sigma[0], tau[0]

    a_quad = 1
    b_quad = -2 * (m**2 * x1 + x1 + m * c)
    c_quad = (m**2 + 1) * (x1**2 + y1**2) - c**2
    discriminant = b_quad**2 - 4 * a_quad * c_quad

    # Handle numerical precision issues
    discriminant = 0.0 if np.isclose(discriminant, 0) else discriminant

    if discriminant >= 0 or np.isclose(discriminant, 0):
        centre_conf = (-b_quad - np.sqrt(discriminant)) / (2 * a_quad)
        radius_conf = np.sqrt((x1 - centre_conf) ** 2 + y1**2)
    else:
        return None

    return m, c, r_unc, centre_conf, radius_conf


def _straight_sections(sigma: np.ndarray, tau: np.ndarray):
    sigma_locus = sigma[1:]
    tau_locus = tau[1:]

    x1, y1 = sigma[0], tau[0]

    m_piece = np.diff(tau_locus) / np.diff(sigma_locus)
    c_piece = tau_locus[:-1] - m_piece * sigma_locus[:-1]

    if c_piece[-1] <= 0:
        raise ValueError("The y-intercept (c) must be positive for the Mohr circle.")

    r_unc = c_piece[-1] / ((m_piece[-1] ** 2 + 1) ** 0.5 - m_piece[-1])

    a_quad = 1
    b_quad = -2 * (m_piece[0] ** 2 * x1 + x1 + m_piece[0] * c_piece[0])
    c_quad = (m_piece[0] ** 2 + 1) * (x1**2 + y1**2) - c_piece[0] ** 2
    discriminant = b_quad**2 - 4 * a_quad * c_quad

    # Handle numerical precision issues
    discriminant = 0.0 if np.isclose(discriminant, 0) else discriminant

    if discriminant >= 0:
        centre_conf = (-b_quad - np.sqrt(discriminant)) / (2 * a_quad)
        radius_conf = np.sqrt((x1 - centre_conf) ** 2 + y1**2)
        return m_piece, c_piece, r_unc, centre_conf, radius_conf
    else:
        return None


def _force_fit(sigma: np.ndarray, tau: np.ndarray) -> tuple:
    sigma_constr, tau_constr = sigma[0], tau[0]
    sigma_locus = sigma[1:]
    tau_locus = tau[1:]

    def linfit(x, m, c):
        return m * x + c

    def f_obj(params):
        m, c = params
        y_pred = linfit(sigma_locus, m, c)
        return np.sum((y_pred - tau_locus) ** 2)

    def preshear_constr(params):
        m, c = params
        tau_pre_pred = linfit(sigma_constr, m, c)
        return tau_pre_pred - tau_constr

    guess = [1.0, 10.0]

    constr = {"type": "eq", "fun": preshear_constr}

    result = optimize.minimize(f_obj, guess, method="SLSQP", constraints=constr)

    m_fit, c_fit = result.x

    r_unc = c_fit / ((m_fit**2 + 1) ** 0.5 - m_fit)

    a_quad = 1.0
    b_quad = -2 * (m_fit**2 * sigma_constr + sigma_constr + m_fit * c_fit)
    c_quad = (m_fit**2 + 1) * (sigma_constr**2 + tau_constr**2) - c_fit**2
    discriminant = b_quad**2 - 4 * a_quad * c_quad

    # Handle numerical precision issues
    discriminant = 0.0 if np.isclose(discriminant, 0) else discriminant

    if discriminant >= 0:
        centre_conf = (-b_quad - np.sqrt(discriminant)) / (2 * a_quad)
        radius_conf = np.sqrt((sigma_constr - centre_conf) ** 2 + tau_constr**2)
        return m_fit, c_fit, r_unc, centre_conf, radius_conf
    else:
        return None


def extract_cell_data():
    """
    Plots Mohr circle and saves data to sqlite database.
    """

    sigma = np.load(os.path.join(outputs_dir, "sigma.npy"))
    tau = np.load(os.path.join(outputs_dir, "shear_stresses.npy"))

    # Only proceed if all shear points are run
    if len(tau) != np.count_nonzero(tau):
        return None

    if (result := _regression_line(sigma, tau)) is not None:
        method = "regression_line"
        m, c, r_unc, centre_conf, radius_conf = result
    elif (result := _straight_sections(sigma, tau)) is not None:
        method = "straight_sections"
        m, c, r_unc, centre_conf, radius_conf = result
    elif (result := _force_fit(sigma, tau)) is not None:
        method = "force_fit"
        m, c, r_unc, centre_conf, radius_conf = result
    else:
        raise RuntimeError("No valid method found for Mohr circle fitting.")

    sigma_c = centre_conf + radius_conf
    sigma_u = r_unc
    ffc = sigma_c / sigma_u
    sigma_fit = np.linspace(0, sigma.max(), 100)

    fig, ax_dict = plt.subplot_mosaic([["A", "A"], ["B", "C"]], layout="constrained")
    phi_i = np.rad2deg(np.arcsin((r_unc - radius_conf) / (r_unc - centre_conf)))

    phi_eff = np.rad2deg(np.arcsin(radius_conf / centre_conf))
    m_eff = np.tan(np.deg2rad(phi_eff))
    tau_eff = m_eff * sigma_fit

    # Plot the Mohr circles
    unc = plt.Circle((r_unc, 0), r_unc, color="black", fill=False)
    conf = plt.Circle((centre_conf, 0), radius_conf, color="black", fill=False)
    ax_dict["A"].add_artist(unc)
    ax_dict["A"].add_artist(conf)

    ax_dict["A"].plot(
        sigma_fit, tau_eff, color="black", linestyle=":", label="Effective Locus"
    )

    if method == "regression_line":
        # Plot sigma vs tau
        ax_dict["A"].scatter(
            sigma[1:], tau[1:], label="Shear points", color="black", marker="o"
        )
        ax_dict["A"].scatter(
            sigma[0], tau[0], label="Pre-shear", color="black", marker="s"
        )

        # And the linear fit
        tau_fit = m * sigma_fit + c
        ax_dict["A"].plot(
            sigma_fit, tau_fit, label="Yield Locus", linestyle="--", color="black"
        )

    elif method == "straight_sections":
        # Plot the piecewise linear fit
        ax_dict["A"].plot(sigma[1:], tau[1:], "o-", label="Shear Points", color="black")
        ax_dict["A"].scatter(
            sigma[0], tau[0], color="black", marker="s", label="Pre-shear"
        )

        # Plot linearised locus
        m_lin_locus = np.tan(np.deg2rad(phi_i))
        c_lin_locus = r_unc * np.sqrt(m_lin_locus**2 + 1) - m_lin_locus * r_unc / 2
        tau_lin_locus = m_lin_locus * sigma_fit + c_lin_locus
        ax_dict["A"].plot(
            sigma_fit,
            tau_lin_locus,
            color="black",
            linestyle="-.",
            label="Linear Locus",
        )

        # Plot the extrapolated confined and unconfined lines
        unc_extr_y = tau[-1] - m[-1] * sigma[-1]
        ax_dict["A"].plot(
            [0, sigma[-1]], [unc_extr_y, tau[-1]], color="black", linestyle="-"
        )
        conf_extr_y = (sigma[0] - sigma[1]) * m[0] + tau[1]
        ax_dict["A"].plot(
            [sigma[1], sigma[0]], [tau[1], conf_extr_y], color="black", linestyle="-"
        )

    elif method == "force_fit":
        ax_dict["A"].scatter(
            sigma[1:], tau[1:], label="Shear points", color="black", marker="o"
        )
        ax_dict["A"].scatter(
            sigma[0], tau[0], label="Pre-shear", color="black", marker="s"
        )

        # Plot the force fit line
        tau_fit = m * sigma_fit + c
        ax_dict["A"].plot(
            sigma_fit, tau_fit, label="Force Fit Locus", linestyle="--", color="black"
        )

    ax_dict["A"].set_aspect("equal", adjustable="box")
    ax_dict["A"].set_xlabel(r"$\sigma$ (Pa)")
    ax_dict["A"].set_ylabel(r"$\tau$ (Pa)")
    ax_dict["A"].set_xlim(0, sigma_c * 1.05)
    ax_dict["A"].set_ylim(0, tau.max() * 1.05)

    row_names = [
        r"$\sigma_{unc}$",
        r"$\sigma_{conf}$",
        r"FFC",
        r"$\phi_{i}$",
        r"$\phi_{eff}$",
        "Fit Method",
    ]
    table_vals = [
        [sigma_c.round(2)],
        [sigma_u.round(2)],
        [ffc.round(2)],
        [phi_i.round(2)],
        [phi_eff.round(2)],
        [method],
    ]

    ax_dict["B"].axis("off")
    tab = ax_dict["B"].table(cellText=table_vals, rowLabels=row_names, loc="top")
    tab.auto_set_column_width(col=0)

    handles, labels = ax_dict["A"].get_legend_handles_labels()
    ax_dict["C"].axis("off")
    ax_dict["C"].legend(handles, labels, loc="upper center", frameon=False)
    fig.tight_layout()

    plt.savefig(os.path.join(outputs_dir, "mohr_circles_plots.png"), dpi=300)
    plt.close()

    json_path = os.path.join(sigma_dir, "../params.json")
    with open(json_path, "r") as f:
        params = json.load(f)

    df = pd.json_normalize(params)
    df.columns = [col.split(".")[-1] if "." in col else col for col in df.columns]
    cols_to_drop = [
        "neighbour_search",
        "n_procs",
        "t_settle",
        "t_compression",
        "t_shear",
        "processor",
    ]
    df.drop(columns=cols_to_drop, inplace=True)
    df["ffc"] = ffc
    df["sigma_unc"] = sigma_c
    df["sigma_conf"] = sigma_u
    df["phi_i"] = phi_i
    df["phi_eff"] = phi_eff
    df["fit_method"] = method

    with sqlite3.connect(os.path.abspath("../../results.db")) as conn:
        df.to_sql("parallel_shear", conn, if_exists="append", index=True)


get_params()
simulate()
postprocess()
extract_cell_data()
