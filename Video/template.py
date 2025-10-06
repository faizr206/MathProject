from manim import *

class EngineSoundVector(Scene):
    def construct(self):
        # ---------- Step 1: Raw waveform ----------
        axes = Axes(
            x_range=[0, 10, 1],
            y_range=[-2, 2, 1],
            x_length=10,
            y_length=3,
            axis_config={"color": WHITE},
        )
        axes.to_edge(UP)
        self.play(Create(axes))

        # Example raw waveform using a sine wave combination
        raw_wave = axes.plot(lambda x: 0.8 * np.sin(2*PI*x) + 0.5*np.sin(5*PI*x), color=YELLOW)
        self.play(Create(raw_wave))
        self.wait(1)

        # Label "Samples per second = vector components"
        zoom_label = Text("Samples per second = vector components", font_size=36, color=BLUE)
        zoom_label.next_to(axes, DOWN, buff=0.5)
        self.play(FadeIn(zoom_label))
        self.wait(1)
        self.play(FadeOut(zoom_label))

        # ---------- Step 2: Fourier Transform (bars in same area) ----------
        # Use same axes to keep bars visible
        freqs = [1, 2.5, 0.8, 0.3, 0.5]
        bars = VGroup()
        for i, amp in enumerate(freqs):
            bar = Rectangle(width=0.8, height=amp, fill_color=ORANGE, fill_opacity=0.7)
            # Place bars starting from y=0 (bottom of waveform axes)
            bar.move_to(axes.c2p(i*2+1, amp/2))  # Spread along x-axis
            bars.add(bar)

        # Animate bars appearing
        self.play(*[GrowFromEdge(bar, edge=DOWN) for bar in bars])
        self.wait(1)

        # ---------- Step 3: Frequency vector ----------
        freq_vector = Text("[f1, f2, f3, ...]", font_size=48, color=GREEN)
        freq_vector.next_to(bars, DOWN, buff=0.5)
        self.play(FadeIn(freq_vector))
        self.wait(2)

        # Optional: highlight connection
        self.play(bars.animate.set_color(TEAL), freq_vector.animate.set_color(TEAL))
        self.wait(2)
