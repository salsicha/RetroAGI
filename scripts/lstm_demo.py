import os
import time


def draw_lstm(frame):
    # Clear terminal screen dynamically
    os.system("cls" if os.name == "nt" else "clear")

    pulse = "■"
    f_valve = "⚡" if frame == 1 else " "
    i_valve = "📥" if frame == 2 else " "
    o_valve = "📦" if frame == 3 else " "

    print("=== LSTM ANIMATED DATA PATH ===")
    print(
        f"Cell State (Long Term):  [Prev C] ───{pulse if frame==0 else '───'}───({f_valve})───{pulse if frame==1 else '───'}───({i_valve})───{pulse if frame==2 or frame==3 else '───'}───► [New C]"
    )
    print("                                         │           ▲")
    print("                                   Forget Gate   Input Gate")
    print("                                         │           │")
    print(
        f"Inputs & Gates:          [Inputs x,h] ───┴───────────┴───────────({o_valve})──► [Output h_t]"
    )
    print("===============================")
    print(
        "\n[Frame Explainer]: "
        + [
            "1. Old Long-Term memory enters the conveyor belt.",
            "2. Forget Gate valve opens! Erasing outdated context.",
            "3. Input Gate valve opens! Pushing new coordinates onto the belt.",
            "4. Output Gate wraps up the state and ships out the short-term prediction.",
        ][frame]
    )


# Run the 4-frame animation loop
for _ in range(3):  # Loop 3 times
    for frame in range(4):
        draw_lstm(frame)
        time.sleep(1.5)  # Wait 1.5 seconds per frame
