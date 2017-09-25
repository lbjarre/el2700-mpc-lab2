import tkinter as tk

class GUI:
    def __init__(self):
        self.root = tk.Tk()

        #self.root.grid()
        self.canvas = tk.Canvas(self.root)
        self.canvas.grid(row=0, column=0)
        #self.root.mainloop()

    def draw_line(self, coords):
        paths = [
            (x_start, y_start, x_end, y_end) for
            (x_start, y_start), (x_end, y_end) in
            zip(coords[0:-2], coords[1:-1])
        ]
        for x_start, y_start, x_end, y_end in paths:
            self.canvas.create_line(x_start, y_start, x_end, y_end)
