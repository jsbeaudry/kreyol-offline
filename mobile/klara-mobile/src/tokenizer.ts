/**
 * Pocket's Kreyòl tokenizer: a Unigram model (sentencepiece-style) read from the tokenizer.json the
 * fine-tune was trained with, so the app produces exactly the ids Python's `tokenizers` does.
 *
 * Its rules, from that file: "▁" goes in front of the text and replaces every space, the text splits
 * before each "▁", each piece is cut by Viterbi over the vocabulary's scores, and a character the
 * vocabulary lacks is spelled out as its UTF-8 bytes (<0xNN> tokens).
 */
type Vocab = [string, number][];

export class UnigramTokenizer {
  private ids = new Map<string, number>();
  private pieces: string[] = [];
  private scores: number[] = [];
  private maxLen = 0;
  private byteIds: number[] = [];
  private unkId: number;
  private unkScore: number;

  constructor(json: { model: { vocab: Vocab; unk_id: number } }) {
    const vocab = json.model.vocab;
    vocab.forEach(([piece, score], id) => {
      this.ids.set(piece, id);
      this.pieces[id] = piece;
      this.scores[id] = score;
      this.maxLen = Math.max(this.maxLen, [...piece].length);
    });
    for (let b = 0; b < 256; b++) {
      this.byteIds[b] = this.ids.get(`<0x${b.toString(16).toUpperCase().padStart(2, '0')}>`) ?? json.model.unk_id;
    }
    this.unkId = json.model.unk_id;
    // sentencepiece's penalty for an unknown character: below the worst real piece.
    this.unkScore = Math.min(...this.scores.filter((s) => s !== 0)) - 10;
  }

  encode(text: string): number[] {
    const spaced = ('▁' + text).replace(/ /g, '▁');
    const words = spaced.split(/(?=▁)/).filter((w) => w.length > 0);
    const out: number[] = [];
    for (const word of words) out.push(...this.encodeWord(word));
    return out;
  }

  /** Back to text: pieces joined, bytes reassembled, "▁" read as a space. */
  decode(ids: number[]): string {
    const bytes: number[] = [];
    let text = '';
    const flush = () => {
      if (bytes.length) text += new TextDecoder().decode(new Uint8Array(bytes));
      bytes.length = 0;
    };
    for (const id of ids) {
      const piece = this.pieces[id] ?? '';
      const byte = /^<0x([0-9A-F]{2})>$/.exec(piece);
      if (byte) bytes.push(parseInt(byte[1], 16));
      else if (!/^<(unk|s|\/s|pad)>$/.test(piece)) { flush(); text += piece; }
    }
    flush();
    return text.replace(/▁/g, ' ').replace(/^ /, '');
  }

  private encodeWord(word: string): number[] {
    const chars = [...word];
    const n = chars.length;
    const best = new Array<number>(n + 1).fill(-Infinity);
    const from = new Array<number>(n + 1).fill(-1);
    const piece = new Array<number>(n + 1).fill(-1);
    best[0] = 0;
    for (let end = 1; end <= n; end++) {
      for (let start = Math.max(0, end - this.maxLen); start < end; start++) {
        if (best[start] === -Infinity) continue;
        const id = this.ids.get(chars.slice(start, end).join(''));
        if (id === undefined) continue;
        const score = best[start] + this.scores[id];
        if (score > best[end]) {
          best[end] = score;
          from[end] = start;
          piece[end] = id;
        }
      }
      // A character with no piece of its own: byte fallback, at the unknown score.
      if (best[end - 1] !== -Infinity && best[end - 1] + this.unkScore > best[end]) {
        best[end] = best[end - 1] + this.unkScore;
        from[end] = end - 1;
        piece[end] = -1;
      }
    }
    const ids: number[] = [];
    for (let end = n; end > 0; end = from[end]) {
      if (piece[end] >= 0) ids.push(piece[end]);
      else {
        const bytes = new TextEncoder().encode(chars[end - 1]);
        for (let i = bytes.length - 1; i >= 0; i--) ids.push(this.byteIds[bytes[i]]);
      }
    }
    return ids.reverse();
  }
}
