"""
Real-Time Fatigue Damage Simulation GUI

A modern, elegant GUI for simulating cumulative fatigue damage in shock absorbers.
Each force input adds to the cumulative damage using Miner's rule.

Author: [Your Name]
License: MIT
"""

import customtkinter as ctk
import numpy as np
from typing import List, Tuple
import math

# Set appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class FatiguePhysicsModel:
    """
    Simplified physics model for the GUI.
    Uses Wöhler curve and Miner's cumulative damage rule.
    """
    
    def __init__(self, 
                 Sut: float = 600,      # Ultimate tensile strength (MPa)
                 d: float = 18,          # Diameter (mm)
                 R: float = 0.95):       # Reliability
        
        self.Sut = Sut
        self.d = d
        self.R = R
        self.D_cumule = 0.0  # Cumulative damage (Miner's rule)
        self.total_cycles = 0
        self.history: List[Tuple[float, float, float, float]] = []  # (force, stress, damage_inc, total_D)
        
        # Pre-calculate fatigue limit
        self.sigma_D = self._corrected_fatigue_limit()
        
    def _corrected_fatigue_limit(self) -> float:
        """Calculate corrected endurance limit."""
        # Simplified correction factors
        Ka = 4.51 * (self.Sut ** -0.265)  # Surface (machined)
        Kb = 1.189 * (self.d ** -0.097)   # Size
        Kc = 0.868                         # Reliability 95%
        return 0.5 * self.Sut * Ka * Kb * Kc
    
    def calculate_stress(self, F: float) -> float:
        """Calculate stress from force."""
        A = np.pi * (self.d / 2) ** 2  # mm²
        Kt = 1.5  # Stress concentration
        return (F / A) * Kt
    
    def cycles_to_failure(self, sigma: float) -> float:
        """Wöhler S-N curve."""
        if sigma <= self.sigma_D:
            return float('inf')
        
        exponent = 3.0 / np.log10((0.9 * self.Sut) / self.sigma_D)
        N = (10**3) * ((0.9 * self.Sut) / sigma) ** exponent
        return max(N, 1)
    
    def apply_force(self, F: float, n_cycles: int = 1) -> Tuple[float, float, float]:
        """
        Apply force and update cumulative damage.
        Returns: (stress, damage_increment, total_damage)
        """
        sigma = self.calculate_stress(F)
        Nf = self.cycles_to_failure(sigma)
        
        if Nf == float('inf'):
            damage_inc = 0.0
        else:
            damage_inc = n_cycles / Nf
        
        self.D_cumule += damage_inc
        self.total_cycles += n_cycles
        
        # Store in history
        self.history.append((F, sigma, damage_inc, self.D_cumule))
        
        return sigma, damage_inc, self.D_cumule
    
    def remaining_life_percentage(self) -> float:
        """Remaining life as percentage (100% = new, 0% = failed)."""
        return max(0, (1 - self.D_cumule) * 100)
    
    def reset(self):
        """Reset damage to zero."""
        self.D_cumule = 0.0
        self.total_cycles = 0
        self.history = []


class DamageGauge(ctk.CTkFrame):
    """Custom circular damage gauge widget."""
    
    def __init__(self, master, size=250, **kwargs):
        super().__init__(master, **kwargs)
        
        self.size = size
        self.damage = 0.0
        
        # Canvas for drawing
        self.canvas = ctk.CTkCanvas(
            self, 
            width=size, 
            height=size, 
            bg='#1a1a2e',
            highlightthickness=0
        )
        self.canvas.pack(padx=10, pady=10)
        
        self.draw_gauge()
    
    def draw_gauge(self):
        """Draw the gauge."""
        self.canvas.delete("all")
        
        cx, cy = self.size // 2, self.size // 2
        r = self.size // 2 - 20
        
        # Background arc (gray)
        self.canvas.create_arc(
            cx - r, cy - r, cx + r, cy + r,
            start=135, extent=270,
            style='arc', width=15,
            outline='#3a3a5c'
        )
        
        # Damage arc (color based on level)
        extent = -270 * min(self.damage, 1.0)
        
        if self.damage < 0.5:
            color = '#00d4aa'  # Green
        elif self.damage < 0.7:
            color = '#ffc107'  # Yellow
        elif self.damage < 0.9:
            color = '#ff9800'  # Orange
        else:
            color = '#ff4444'  # Red
        
        if extent != 0:
            self.canvas.create_arc(
                cx - r, cy - r, cx + r, cy + r,
                start=135, extent=extent,
                style='arc', width=15,
                outline=color
            )
        
        # Center text - percentage
        percentage = min(self.damage * 100, 100)
        self.canvas.create_text(
            cx, cy - 10,
            text=f"{percentage:.1f}%",
            font=("Helvetica", 32, "bold"),
            fill='white'
        )
        
        # Label
        self.canvas.create_text(
            cx, cy + 30,
            text="DAMAGE",
            font=("Helvetica", 12),
            fill='#888888'
        )
        
        # Status text
        if self.damage >= 1.0:
            status = "⚠️ FAILED"
            status_color = '#ff4444'
        elif self.damage >= 0.9:
            status = "CRITICAL"
            status_color = '#ff4444'
        elif self.damage >= 0.7:
            status = "WARNING"
            status_color = '#ff9800'
        else:
            status = "NORMAL"
            status_color = '#00d4aa'
        
        self.canvas.create_text(
            cx, cy + 55,
            text=status,
            font=("Helvetica", 14, "bold"),
            fill=status_color
        )
    
    def set_damage(self, damage: float):
        """Update damage value and redraw."""
        self.damage = damage
        self.draw_gauge()


class DamageSimulationApp(ctk.CTk):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        
        # Window setup
        self.title("🔧 Shock Absorber Fatigue Monitor")
        self.geometry("600x750")
        self.resizable(False, False)
        
        # Physics model
        self.model = FatiguePhysicsModel()
        
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        
        # Create UI components
        self._create_header()
        self._create_gauge()
        self._create_stats()
        self._create_input()
        self._create_history()
        
    def _create_header(self):
        """Create header section."""
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, pady=(20, 10), sticky="ew")
        
        title = ctk.CTkLabel(
            header,
            text="⚙️ SHOCK ABSORBER FATIGUE MONITOR",
            font=("Helvetica", 20, "bold")
        )
        title.pack()
        
        subtitle = ctk.CTkLabel(
            header,
            text="Real-Time Cumulative Damage Simulation (Wöhler + Miner)",
            font=("Helvetica", 12),
            text_color="#888888"
        )
        subtitle.pack()
    
    def _create_gauge(self):
        """Create damage gauge."""
        self.gauge = DamageGauge(self)
        self.gauge.grid(row=1, column=0, pady=10)
    
    def _create_stats(self):
        """Create statistics display."""
        stats_frame = ctk.CTkFrame(self)
        stats_frame.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        stats_frame.grid_columnconfigure((0, 1, 2), weight=1)
        
        # Remaining life
        life_frame = ctk.CTkFrame(stats_frame, fg_color="#2a2a4a")
        life_frame.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        
        ctk.CTkLabel(
            life_frame,
            text="Remaining Life",
            font=("Helvetica", 10),
            text_color="#888888"
        ).pack(pady=(5, 0))
        
        self.life_label = ctk.CTkLabel(
            life_frame,
            text="100.0%",
            font=("Helvetica", 18, "bold"),
            text_color="#00d4aa"
        )
        self.life_label.pack(pady=(0, 5))
        
        # Total cycles
        cycles_frame = ctk.CTkFrame(stats_frame, fg_color="#2a2a4a")
        cycles_frame.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        ctk.CTkLabel(
            cycles_frame,
            text="Total Cycles",
            font=("Helvetica", 10),
            text_color="#888888"
        ).pack(pady=(5, 0))
        
        self.cycles_label = ctk.CTkLabel(
            cycles_frame,
            text="0",
            font=("Helvetica", 18, "bold")
        )
        self.cycles_label.pack(pady=(0, 5))
        
        # Last stress
        stress_frame = ctk.CTkFrame(stats_frame, fg_color="#2a2a4a")
        stress_frame.grid(row=0, column=2, padx=5, pady=5, sticky="ew")
        
        ctk.CTkLabel(
            stress_frame,
            text="Last Stress",
            font=("Helvetica", 10),
            text_color="#888888"
        ).pack(pady=(5, 0))
        
        self.stress_label = ctk.CTkLabel(
            stress_frame,
            text="0 MPa",
            font=("Helvetica", 18, "bold")
        )
        self.stress_label.pack(pady=(0, 5))
    
    def _create_input(self):
        """Create input controls."""
        input_frame = ctk.CTkFrame(self)
        input_frame.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        input_frame.grid_columnconfigure(1, weight=1)
        
        # Force input
        ctk.CTkLabel(
            input_frame,
            text="Force (N):",
            font=("Helvetica", 14)
        ).grid(row=0, column=0, padx=(10, 5), pady=10)
        
        self.force_entry = ctk.CTkEntry(
            input_frame,
            width=150,
            font=("Helvetica", 14),
            placeholder_text="e.g., 15000"
        )
        self.force_entry.grid(row=0, column=1, padx=5, pady=10, sticky="w")
        self.force_entry.insert(0, "15000")
        self.force_entry.bind("<Return>", lambda e: self.apply_force())
        
        # Cycles input
        ctk.CTkLabel(
            input_frame,
            text="Cycles:",
            font=("Helvetica", 14)
        ).grid(row=0, column=2, padx=(20, 5), pady=10)
        
        self.cycles_entry = ctk.CTkEntry(
            input_frame,
            width=80,
            font=("Helvetica", 14),
            placeholder_text="1"
        )
        self.cycles_entry.grid(row=0, column=3, padx=5, pady=10)
        self.cycles_entry.insert(0, "1000")
        
        # Buttons
        btn_frame = ctk.CTkFrame(input_frame, fg_color="transparent")
        btn_frame.grid(row=1, column=0, columnspan=4, pady=10)
        
        self.apply_btn = ctk.CTkButton(
            btn_frame,
            text="⚡ APPLY FORCE",
            font=("Helvetica", 14, "bold"),
            width=150,
            height=40,
            fg_color="#00aa88",
            hover_color="#008866",
            command=self.apply_force
        )
        self.apply_btn.pack(side="left", padx=10)
        
        self.reset_btn = ctk.CTkButton(
            btn_frame,
            text="🔄 RESET",
            font=("Helvetica", 14),
            width=100,
            height=40,
            fg_color="#555555",
            hover_color="#444444",
            command=self.reset
        )
        self.reset_btn.pack(side="left", padx=10)
    
    def _create_history(self):
        """Create history log."""
        history_frame = ctk.CTkFrame(self)
        history_frame.grid(row=4, column=0, padx=20, pady=10, sticky="nsew")
        self.grid_rowconfigure(4, weight=1)
        
        ctk.CTkLabel(
            history_frame,
            text="📋 Force History",
            font=("Helvetica", 14, "bold")
        ).pack(anchor="w", padx=10, pady=(10, 5))
        
        self.history_text = ctk.CTkTextbox(
            history_frame,
            font=("Consolas", 11),
            height=120
        )
        self.history_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.history_text.configure(state="disabled")
    
    def apply_force(self):
        """Apply force and update display."""
        try:
            force = float(self.force_entry.get())
            cycles = int(self.cycles_entry.get())
            
            if force <= 0 or cycles <= 0:
                raise ValueError("Values must be positive")
            
        except ValueError as e:
            self._show_error(str(e))
            return
        
        # Apply force to model
        stress, damage_inc, total_D = self.model.apply_force(force, cycles)
        
        # Update gauge
        self.gauge.set_damage(total_D)
        
        # Update stats
        remaining = self.model.remaining_life_percentage()
        self.life_label.configure(
            text=f"{remaining:.1f}%",
            text_color=self._get_color(remaining / 100)
        )
        self.cycles_label.configure(text=f"{self.model.total_cycles:,}")
        self.stress_label.configure(text=f"{stress:.1f} MPa")
        
        # Update history
        self._add_history(force, cycles, stress, damage_inc, total_D)
        
        # Check failure
        if total_D >= 1.0:
            self.apply_btn.configure(state="disabled")
            self._show_failure_alert()
    
    def _get_color(self, value: float) -> str:
        """Get color based on remaining life."""
        if value > 0.5:
            return "#00d4aa"
        elif value > 0.3:
            return "#ffc107"
        elif value > 0.1:
            return "#ff9800"
        else:
            return "#ff4444"
    
    def _add_history(self, force, cycles, stress, damage_inc, total_D):
        """Add entry to history."""
        self.history_text.configure(state="normal")
        
        entry = f"F={force:,.0f}N × {cycles:,} cycles → σ={stress:.1f}MPa → ΔD={damage_inc:.6f} → D={total_D:.4f} ({total_D*100:.2f}%)\n"
        
        self.history_text.insert("1.0", entry)
        self.history_text.configure(state="disabled")
    
    def _show_failure_alert(self):
        """Show failure alert."""
        alert = ctk.CTkToplevel(self)
        alert.title("⚠️ Component Failure")
        alert.geometry("300x150")
        alert.transient(self)
        alert.grab_set()
        
        ctk.CTkLabel(
            alert,
            text="⚠️ FAILURE",
            font=("Helvetica", 24, "bold"),
            text_color="#ff4444"
        ).pack(pady=20)
        
        ctk.CTkLabel(
            alert,
            text="Cumulative damage has exceeded 100%.\nReplace the shock absorber!",
            font=("Helvetica", 12)
        ).pack()
        
        ctk.CTkButton(
            alert,
            text="OK",
            command=alert.destroy
        ).pack(pady=15)
    
    def _show_error(self, message: str):
        """Show error message."""
        alert = ctk.CTkToplevel(self)
        alert.title("Error")
        alert.geometry("250x100")
        alert.transient(self)
        
        ctk.CTkLabel(alert, text=message).pack(pady=20)
        ctk.CTkButton(alert, text="OK", command=alert.destroy).pack()
    
    def reset(self):
        """Reset the simulation."""
        self.model.reset()
        self.gauge.set_damage(0)
        self.life_label.configure(text="100.0%", text_color="#00d4aa")
        self.cycles_label.configure(text="0")
        self.stress_label.configure(text="0 MPa")
        self.apply_btn.configure(state="normal")
        
        self.history_text.configure(state="normal")
        self.history_text.delete("1.0", "end")
        self.history_text.configure(state="disabled")


def main():
    """Run the application."""
    app = DamageSimulationApp()
    app.mainloop()


if __name__ == "__main__":
    main()
