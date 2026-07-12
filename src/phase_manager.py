"""
PhaseManager - Co che chuyen pha CASL tu dong.

Pha I (Communication-efficient):
  - Bottleneck bat -> nen bieu dien truyen qua mang
  - Client frozen, chi server LoRA duoc cap nhat

Pha II (Standard split):
  - Bottleneck tat
  - LoRA ca 2 phia (client + server)

Trigger: Quan sat p round gan nhat. Neu tat ca p loss
         deu nho hon nguong thresh_val_loss -> chuyen sang Pha II.
"""

from collections import deque


class PhaseManager:
    def __init__(self, thresh_val_loss: float = 0.01, window: int = 3, enable: bool = True):
        self.thresh = thresh_val_loss
        self.window = window
        self.enable = enable
        self.phase = 1
        self._loss_window: deque = deque(maxlen=window) # Cua so truot luu val_loss cua p round gan nhat

    @property
    def current_phase(self) -> int:
        return self.phase

    def is_phase1(self) -> bool:
        return self.phase == 1

    def is_phase2(self) -> bool:
        return self.phase == 2

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def update(self, val_loss: float) -> bool:
        """
        Cap nhat validation loss va kiem tra trigger chuyen pha.

        Trigger: Neu da quan sat du `window` round VA tat ca loss trong
                 cua so deu < thresh_val_loss -> chuyen Pha I -> Pha II.

        Args:
            val_loss: Validation loss sau round hien tai.

        Returns:
            True neu vua chuyen tu Pha I -> Pha II trong lan goi nay.
            False trong moi truong hop khac (chua du round, da o Pha II,
            hoac enable=False).
        """
        # Neu co che chuyen pha bi tat, khong lam gi ca
        if not self.enable:
            return False

        # Da o Pha II, chi ghi nhan loss de log, khong chuyen nua
        if self.phase == 2:
            self._loss_window.append(val_loss)
            return False

        # Them loss moi vao cua so truot
        self._loss_window.append(val_loss)

        # Kiem tra: da du p round va toan bo loss < nguong
        if len(self._loss_window) == self.window:
            all_below = all(loss < self.thresh for loss in self._loss_window)
            if all_below:
                self.phase = 2
                return True

        return False

    def apply_phase2_config(self, fine_tune_config: dict, bottleneck_config: dict):
        """
        Cap nhat fine_tune_config va bottleneck_config sang cau hinh Pha II:
          - bottleneck.enable = False  (go bo khoi co chai)
          - fine-tune.client = True    (mo khoa LoRA phia client)
          - fine-tune.server = True    (giu nguyen LoRA phia server)
        """
        bottleneck_config["enable"] = False
        fine_tune_config["client"] = True
        fine_tune_config["server"] = True

        _HEADER = '\033[95m'
        _BOLD   = '\033[1m'
        _END    = '\033[0m'
        losses_str = ", ".join(f"{l:.4f}" for l in self._loss_window)
        banner = (
            f"\n{_HEADER}{_BOLD}"
            f"{'=' * 60}\n"
            f"  [CASL] CHUYEN PHA THANH CONG: PHA I  -->  PHA II\n"
            f"  TRIGGER: {self.window} round gan nhat [{losses_str}]\n"
            f"           tat ca < nguong {self.thresh:.4f}\n"
            f"  BOTTLENECK: OFF  |  LORA CLIENT: ON  |  LORA SERVER: ON\n"
            f"{'=' * 60}"
            f"{_END}\n"
        )
        print(banner)

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------

    def status_str(self) -> str:
        """Chuoi trang thai ngan gon de log."""
        losses = list(self._loss_window)
        return (
            f"PhaseManager | phase={self.phase} | enable={self.enable} | "
            f"window={self.window} | thresh={self.thresh} | "
            f"recent_losses={losses}"
        )
