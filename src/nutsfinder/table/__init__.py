"""A local, browser-based poker table: play against a bot, or replay bot-vs-bot hands.

- :mod:`nutsfinder.table.session`: the game logic, no HTTP. ``PlaySession``
  drives one human-vs-bot hand at a time with the engine's ``new_hand`` /
  ``apply_action`` / ``deal_board`` (the bot is asked and told exactly what
  :func:`nutsfinder.engine.runner.play_hand` would ask and tell it);
  ``ReplayJob`` plays bots against each other with ``play_hand`` itself and
  keeps every decision's debug record.
- :mod:`nutsfinder.table.server`: a stdlib ``ThreadingHTTPServer`` serving
  ``index.html`` (one self-contained page) and a small JSON API.

Run it with ``.venv\\Scripts\\python examples\\poker_table.py``.
"""
