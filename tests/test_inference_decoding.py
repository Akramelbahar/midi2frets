"""Structured fretboard decoding (§7): string occupancy state (a string still
ringing from an earlier note cannot be re-attacked by a new note) and its
effect on beam search, isolated from any technique decoding concerns."""
import torch

from inference import string_free_at, beam_search_predict

TUNING = [64, 59, 55, 50, 45, 40]


def _note(pitch, time, dur_ticks=240):
    return {"pitch": pitch, "time": time, "dur_ticks": dur_ticks}


def test_string_free_at_tracks_last_note_end_per_string():
    notes = [_note(64, 0, dur_ticks=1000), _note(64, 100, dur_ticks=50), _note(59, 200, dur_ticks=10)]
    seq = [0, 1, 0]  # note0->string0, note1->string1, note2->string0
    free_at = string_free_at(notes, seq, up_to_i=3)
    assert free_at[0] == 1000  # note0 (0..1000) overwritten only if a LATER note on string0 ends later; note2 ends at 210
    assert free_at[1] == 150   # note1: 100+50
    assert free_at[2] == 0     # never used


def test_string_free_at_empty_prefix_is_all_free():
    notes = [_note(64, 0, dur_ticks=1000)]
    assert string_free_at(notes, [0], up_to_i=0) == [0] * 6


class _FakeStringModel:
    """Always strongly prefers string 0, then string 1, over any other
    string -- deterministic enough to prove whether the occupancy
    constraint actually changes the decoded string for an overlapping note."""

    def eval(self):
        pass

    def __call__(self, features, pad_mask):
        B, T = pad_mask.shape
        logits = torch.zeros(B, T, 6)
        logits[..., 0] = 10.0
        logits[..., 1] = 5.0
        return logits


def test_beam_search_avoids_reattacking_a_still_ringing_string():
    # note0 sustains from 0 to 1000; note1 starts at 500, well inside note0's
    # sustain. Both pitches are playable on every string (pitch=64 fits
    # frets 0..24 on all 6 strings at this tuning), so without the occupancy
    # constraint the model's strong string-0 preference would put BOTH notes
    # on string 0 -- physically impossible (string 0 is still ringing).
    notes = [_note(64, 0, dur_ticks=1000), _note(64, 500, dur_ticks=100)]
    model = _FakeStringModel()
    preds = beam_search_predict(model, notes, TUNING, capo=0, beam_width=3)
    assert preds[0] == 0, "first note should still take the model's strongly preferred string 0"
    assert preds[1] != 0, "second note must NOT reattack string 0 while it is still ringing"
    assert preds[1] == 1, "next-best string (1) should be chosen instead"


def test_beam_search_allows_reattack_after_string_is_free():
    # note1 now starts AFTER note0's sustain ends (1000) -- string 0 is free
    # again, so both notes CAN legitimately land on string 0.
    notes = [_note(64, 0, dur_ticks=1000), _note(64, 1000, dur_ticks=100)]
    model = _FakeStringModel()
    preds = beam_search_predict(model, notes, TUNING, capo=0, beam_width=3)
    assert preds[0] == 0
    assert preds[1] == 0


def test_beam_search_still_allows_simultaneous_chord_notes_on_different_strings():
    # Sanity check the occupancy constraint doesn't interfere with the
    # pre-existing same-onset chord rule.
    notes = [_note(64, 0, dur_ticks=240), _note(67, 0, dur_ticks=240)]
    model = _FakeStringModel()
    preds = beam_search_predict(model, notes, TUNING, capo=0, beam_width=3)
    assert preds[0] != preds[1]
