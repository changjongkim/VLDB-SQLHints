
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def draw_detailed_halo_logic():
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Colors
    c_blue = '#0288D1'   # Data/Input
    c_orange = '#F57C00' # Processing
    c_purple = '#7E57C2' # AI/Sigma
    c_green = '#388E3C'  # Output/Policy
    c_bg = '#FAFAFA'
    
    # Background
    ax.add_patch(patches.Rectangle((0, 0), 100, 100, color=c_bg, zorder=0))

    # --- Phase 1: Input & Transformation ---
    ax.text(5, 92, "PHASE I: DATA FEATURIZATION", fontweight='bold', fontsize=14, color=c_blue)
    # Box 1: SQL & HW
    ax.add_patch(patches.FancyBboxPatch((5, 75), 20, 12, boxstyle="round,pad=1", ec=c_blue, fc='white', lw=2))
    ax.text(15, 83, "Inputs\n- SQL Query\n- HW Profiles (A, B)", ha='center', va='center', fontsize=10)
    
    # Arrow
    ax.annotate("", xy=(30, 81), xytext=(25, 81), arrowprops=dict(arrowstyle="->", lw=2, color='gray'))
    
    # Box 2: Extraction
    ax.add_patch(patches.FancyBboxPatch((30, 75), 25, 12, boxstyle="round,pad=1", ec=c_orange, fc='white', lw=2))
    ax.text(42.5, 81, "Feature Engineering\n- Op Type One-hot\n- Cost/Self-time Ratios\n- ΔHW Binary Flags", ha='center', va='center', fontsize=10)

    # --- Phase 2: The Sigma预测 Engine ---
    ax.text(5, 65, "PHASE II: OPERATOR-LEVEL σ PREDICTION", fontweight='bold', fontsize=14, color=c_purple)
    
    # AI Brain Icon (Simplified)
    ax.add_patch(patches.Circle((15, 45), 8, fc=c_purple, alpha=0.2, ec=c_purple, lw=2))
    ax.text(15, 45, "RF\nσ-Model", ha='center', va='center', fontsize=12, fontweight='bold', color=c_purple)

    # Predictions Flow
    for i, op in enumerate(['Scan', 'Join', 'Sort']):
        y_pos = 55 - (i * 10)
        ax.annotate("", xy=(38, y_pos), xytext=(23, 45), arrowprops=dict(arrowstyle="->", lw=1, color=c_purple, alpha=0.5))
        ax.add_patch(patches.Rectangle((38, y_pos-3), 20, 6, fc='white', ec=c_purple, lw=1))
        ax.text(48, y_pos, f"Node_{i}: Predicted Δ{op}", ha='center', va='center', fontsize=9)

    # --- Phase 3: Risk Aggregation & Policy (The Hub) ---
    ax.text(65, 92, "PHASE III: ROBUST POLICY", fontweight='bold', fontsize=14, color=c_green)
    
    # The Decision Container
    ax.add_patch(patches.FancyBboxPatch((65, 35), 32, 52, boxstyle="round,pad=1", ec=c_green, fc='white', lw=3))
    
    # Logic 1: Weighted Sum
    ax.add_patch(patches.Rectangle((68, 75), 26, 8, fc=c_green, alpha=0.1, ec=c_green))
    ax.text(81, 79, "Risk Aggregation\nΣ(Δ_op * log(self_time))", ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Logic 2: EV Calculation
    ax.add_patch(patches.Rectangle((68, 62), 26, 8, fc=c_green, alpha=0.1, ec=c_green))
    ax.text(81, 66, "Expected Value (EV)\nProb(Reg) vs. Speedup", ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Logic 3: 3-Tier Classifier
    y_tier = 38
    ax.text(81, 54, "Selection Hub", ha='center', fontweight='bold', fontsize=10)
    ax.add_patch(patches.Rectangle((68, 38), 26, 12, fc='white', ec='gray', ls='--'))
    ax.text(72, 47, "G", color='green', fontweight='bold', fontsize=12)
    ax.text(72, 44, "Y", color='orange', fontweight='bold', fontsize=12)
    ax.text(72, 41, "R", color='red', fontweight='bold', fontsize=12)
    ax.text(83, 44, "1. Green -> Apply\n2. Yellow -> EV Filter\n3. Red -> Fallback", fontsize=8)

    # Final Execution Arrow
    ax.annotate("Recommended\nHint Set", xy=(81, 15), xytext=(81, 35), 
                arrowprops=dict(facecolor=c_green, shrink=0.05, width=10, headwidth=20))

    # Legend / Info
    ax.add_patch(patches.Rectangle((5, 5), 50, 20, fc='white', ec='black', alpha=0.7))
    ax.text(7, 21, "Notation:", fontweight='bold')
    ax.text(7, 17, "- Δ_op: Operator performance shift on target HW", fontsize=9)
    ax.text(7, 13, "- σ Model: Pre-trained Random Forest on hardware delta", fontsize=9)
    ax.text(7, 9, "- EV: Expected Benefit considering prediction uncertainty", fontsize=9)

    plt.suptitle("HALO: Detailed Analytical Procedure & Data Flow Diagram", fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig("/root/halo/results/figures/halo_detailed_procedure.png", dpi=300, bbox_inches='tight')
    print("Detailed procedure diagram saved to /root/halo/results/figures/halo_detailed_procedure.png")

if __name__ == "__main__":
    draw_detailed_halo_logic()
