import sys
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt

class SpriteViewer(QLabel):
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        pixmap = QPixmap(image_path)
        self.setPixmap(pixmap)
        self.setFixedSize(pixmap.size())

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sprites Separati con PyQt6")

        # Ottieni le dimensioni dello schermo principale
        screen_size = app.primaryScreen().size()
        
        # Imposta la geometria della finestra a schermo intero
        self.setGeometry(0, 0, screen_size.width(), screen_size.height())

        # ---
        # Crea il primo sfondo, lo ridimensiona e lo rende visibile
        initial_background_pixmap = QPixmap('sfondo1.png')
        if not initial_background_pixmap.isNull():
            initial_background_pixmap = initial_background_pixmap.scaled(
                screen_size,
                Qt.AspectRatioMode.IgnoreAspectRatio,  # Ignora l'aspect ratio
                Qt.TransformationMode.SmoothTransformation
            )
        self.initial_background = QLabel(self)
        self.initial_background.setPixmap(initial_background_pixmap)
        self.initial_background.setGeometry(0, 0, screen_size.width(), screen_size.height())

        # ---
        # Crea il secondo sfondo ('tamplate.png'), lo ridimensiona e lo nasconde
        template_background_pixmap = QPixmap('tamplate.png')
        if not template_background_pixmap.isNull():
            template_background_pixmap = template_background_pixmap.scaled(
                screen_size,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        self.template_background = QLabel(self)
        self.template_background.setPixmap(template_background_pixmap)
        self.template_background.setGeometry(0, 0, screen_size.width(), screen_size.height())
        self.template_background.hide()
        
        # ---
        # Crea gli altri sprite, che hanno una dimensione fissa, e li nasconde
        self.sprite1 = SpriteViewer('press-space.png', self)
        self.sprite1.move(50, 50)
        self.sprite1.hide()

        self.sprite2 = SpriteViewer('cut.png', self)
        self.sprite2.move(500, 50)
        self.sprite2.hide()

        self.sprite3 = SpriteViewer('chi ki.png', self)
        self.sprite3.move(250, 50)
        self.sprite3.hide()
        
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self.initial_background.hide()
            self.template_background.show()
            self.sprite1.show()
            self.sprite2.show()
            self.sprite3.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.showFullScreen()
    sys.exit(app.exec())