
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_halo_procedure():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Color Palette
    color_src = '#E1F5FE'  # Blue
    color_ml = '#F3E5F5'   # Purple
    color_tgt = '#E8F5E9'  # Green
    color_arrow = '#37474F'

    # 1. Source Environment Section
    ax.add_patch(patches.FancyBboxPatch((2, 60), 28, 30, boxstyle="round,pad=2", ec="gray", fc=color_src, alpha=0.5))
    ax.text(16, 92, "1. Source Environment (Profiling)", ha='center', fontsize=12, fontweight='bold')
    
    # Sub-components of Source
    comp_src = [("Query Profiler", 80), ("Hint Space", 72), ("Execution Traces", 64)]
    for text, y in comp_src:
        ax.add_patch(patches.Rectangle((5, y), 22, 5, fc='white', ec='gray', lw=1, zorder=2))
        ax.text(16, y+2, text, ha='center', va='center', fontsize=10)

    # 2. IA / Feature Extraction (Transition)
    ax.annotate("", xy=(38, 75), xytext=(30, 75), arrowprops=dict(arrowstyle="->", lw=2, color=color_arrow))
    ax.text(34, 77, "Plan & HW\nMetrics", ha='center', fontsize=9)

    # 3. Intelligence Section (The Core)
    ax.add_patch(patches.FancyBboxPatch((38, 40), 28, 50, boxstyle="round,pad=2", ec="#7E57C2", fc=color_ml, alpha=0.5))
    ax.text(52, 92, "2. HALO Intelligence (AI Engine)", ha='center', fontsize=12, fontweight='bold', color="#5E35B1")
    
    ax.add_patch(patches.Rectangle((41, 75), 22, 10, fc='#7E57C2', ec='white', lw=1, alpha=0.8))
    ax.text(52, 80, "σ Model Learning\n(Random Forest)", ha='center', va='center', color='white', fontweight='bold')
    
    ax.add_patch(patches.Rectangle((41, 55), 22, 8, fc='white', ec='#7E57C2', lw=1))
    ax.text(52, 59, "Operator-Level\nRisk Assessment", ha='center', va='center', fontsize=10)
    
    ax.add_patch(patches.Rectangle((41, 44), 22, 8, fc='white', ec='#7E57C2', lw=1))
    ax.text(52, 48, "Risk-Reward\nTradeoff Logic", ha='center', va='center', fontsize=10)

    # 4. Target Environment Section
    ax.annotate("", xy=(74, 75), xytext=(66, 75), arrowprops=dict(arrowstyle="->", lw=2, color=color_arrow))
    ax.text(70, 77, "Safe\nHints", ha='center', fontsize=9)

    ax.add_patch(patches.FancyBboxPatch((74, 60), 24, 30, boxstyle="round,pad=2", ec="#2E7D32", fc=color_tgt, alpha=0.5))
    ax.text(86, 92, "3. Target Environment", ha='center', fontsize=12, fontweight='bold', color="#2E7D32")
    
    comp_tgt = [("Policy Controller", 80), ("Final Hint Selection", 72), ("Native Fallback", 64)]
    for text, y in comp_tgt:
        ax.add_patch(patches.Rectangle((77, y), 18, 5, fc='white', ec='#2E7D32', lw=1, zorder=2))
        ax.text(86, y+2, text, ha='center', va='center', fontsize=10)

    # 5. Legend for Policy Tiers (Bottom)
    ax.add_patch(patches.Rectangle((38, 10), 28, 20, fc='white', ec='gray', ls='--', alpha=0.5))
    ax.text(52, 26, "HALO-R Decision Tiers", ha='center', fontweight='bold', fontsize=10)
    ax.add_patch(patches.Circle((42, 21), 1.5, color='green', alpha=0.6))
    ax.text(45, 20, "Green: Safe & High Gain", fontsize=9)
    ax.add_patch(patches.Circle((42, 17), 1.5, color='orange', alpha=0.6))
    ax.text(45, 16, "Yellow: Uncertain (Calc EV)", fontsize=9)
    ax.add_patch(patches.Circle((42, 13), 1.5, color='red', alpha=0.6))
    ax.text(45, 12, "Red: Danger (Fallback)", fontsize=9)

    plt.title("HALO Framework: System Components & Procedure Workflow", fontsize=16, fontweight='bold', pad=20)
    plt.savefig("/root/halo/results/figures/halo_procedure_component.png", dpi=300, bbox_inches='tight')
    print("Procedure diagram saved to /root/halo/results/figures/halo_procedure_component.png")

if __name__ == "__main__":
    draw_halo_procedure()
