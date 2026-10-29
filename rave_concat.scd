/*
Concatenative synthesis test: one value between 0 and 1 controls which parts of a real recording are played.

How it works
- make_corpus.py cuts a recording into overlapping segments (0.5, 1, 2, 3 or 5 s) and measures each one:
  loudness, hiss, brightness, rumble, activity, noisiness, and the number of sound events per second.
  The tables are saved next to the recording (name_seg1.csv and so on).
- Each feature has a weight from -1 to 1 (a preset sets them; the sliders change them). Every feature is first
  turned into a rank between 0 and 1, so all features count equally, then multiplied by its weight and added.
  The segments are ordered by this sum: rank 0 = the lowest, rank 1 = the highest (or the reverse, with "reverse").
  A negative weight means that more of that feature pushes towards value 0.
- The preset "wash to crash" comes from the analysis of Seawaves1.wav: in that recording the hissy, steady
  stretches (wash) and the rumbly, bursty ones (crashes) are the main contrast. Other recordings need other weights.
- The text under the picture shows how different the two ends are (the lowest and the highest 10 % of the
  segments): it gives the mean loudness, hiss, activity and event density of each end. If those numbers are
  close, the recording itself has little variation along this axis, and no setting can fix that.
- The value picks a rank. A routine plays one segment after the other, each one crossfaded into the next,
  always choosing a segment near the current value. All sound is the original recording, nothing is generated.
- "spread" is how wide the choice is around the value (as a fraction of all segments). A small spread follows
  the value closely, a larger one gives more variety. Segments that were played recently, or that overlap in time
  with them, are skipped, so the same piece is not repeated at once.
- Response time: a new value is used when the next segment starts, so the delay is about the segment length minus
  the crossfade. Short segments react faster; longer ones keep calls and splashes together.

Use your own recordings
- Put the file (wav, flac or aiff) in the audio folder and press "rescan folder", or use "choose another file...".
- When a file has no table yet, the window runs make_corpus.py for it (a few seconds for a minute of audio) and
  loads the result. "analyse again" does the same by hand. This needs the Python in .venv with numpy, scipy
  and soundfile (the same one that makes the sound maps).

The picture shows every segment: left to right is its position in the recording, up is its rank for the
chosen feature. Blue = candidates for the current value, orange = recently played, ring = playing now,
the horizontal line = the value.

From code:  ~ccSet.(0.3);   // sets the value (and moves the slider)

I could not run SuperCollider, so this is untested. If something fails, send me the post window text.
*/

// 1. Setup and window (run once; run again to rebuild)
(
s.waitForBoot {
	~ccDir      = "/Users/marcellolussana/Documents/Software/RAVE_SC/";
	~ccAudioDir = ~ccDir ++ "audio/";
	~ccPy       = ~ccDir ++ ".venv/bin/python";
	~ccScript   = ~ccDir ++ "make_corpus.py";
	// segment length in hundredths of a second (slider: 10 .. 100 = 0.1 .. 1 s, steps of 0.05 s)
	~ccLenCs = 100;
	~ccLenStr = { |cs|   // the text used in the table's file name, same as python's {:g}: 10 -> "0.1", 15 -> "0.15", 100 -> "1"
		var fr = (cs % 100).asString, t;
		if(fr.size < 2) { fr = "0" ++ fr };
		t = (cs div: 100).asString ++ "." ++ fr;
		while { t.last == $0 } { t = t.drop(-1) };
		if(t.last == $.) { t = t.drop(-1) };
		t
	};
	// name, column in the csv (start, dur, rms_db, centroid, flatness, low_share, activity, centroid_hi, hf_db, onset_rate, timbre1, timbre2)
	~ccFeatures = [
		["loudness (rms dB)", 2],
		["amount of hiss (energy above 2 kHz)", 8],
		["brightness of the hiss (centroid above 500 Hz)", 7],
		["rumble (share of energy below 300 Hz)", 5],
		["activity (loudness variation: calm to bursty)", 6],
		["noisiness (spectral flatness)", 4],
		["overall brightness (centroid)", 3],
		["event density (sounds per second)", 9],
		["timbre axis 1 (MFCC, main spectral-shape difference in this recording)", 10],
		["timbre axis 2 (MFCC, second spectral-shape difference)", 11]
	];
	~ccQ = { |p| "\"" ++ p ++ "\"" };

	// segment players, stereo or mono file. Equal-power crossfade (square root of a linear fade).
	SynthDef(\ccSeg2, { |buf, start = 0, dur = 1, xf = 0.3, amp = 1, out = 0, gbus = 0, gidx = 0|
		var env = EnvGen.kr(Env([0, 1, 1, 0], [xf, (dur - (2 * xf)).max(0.001), xf]), doneAction: 2).sqrt;
		var sig = PlayBuf.ar(2, buf, BufRateScale.kr(buf), startPos: start);
		Out.ar(out, sig * env * amp * In.kr(gbus + gidx).lag(0.05));
	}).add;
	SynthDef(\ccSeg1, { |buf, start = 0, dur = 1, xf = 0.3, amp = 1, out = 0, gbus = 0, gidx = 0|
		var env = EnvGen.kr(Env([0, 1, 1, 0], [xf, (dur - (2 * xf)).max(0.001), xf]), doneAction: 2).sqrt;
		var sig = PlayBuf.ar(1, buf, BufRateScale.kr(buf), startPos: start);
		Out.ar(out, Pan2.ar(sig * env * amp * In.kr(gbus + gidx).lag(0.05)));
	}).add;
	s.sync;

	~ccFile = nil;
	~ccTried = nil;
	~ccBuf = nil;
	~ccSR = 44100; ~ccCh = 2; ~ccFileDur = nil;
	~ccVal = 0.5;
	~ccW = [0, -1, 0, 1, 1, 0, 0, 1, 0, 0];   // weight of each feature in ~ccFeatures (-1 .. 1): wash to crash
	~ccNormRank = [];
	~ccLenToken = 0;
	~ccRev = false;
	~ccSpread = 0.05;
	~ccXf = 0.3;
	~ccAmp = 0.7;
	~ccPlaying = false;
	~ccR = nil;
	~ccSweepR = nil;
	~ccRecent = [];
	~ccMode = 0;            // 0 = ordered by features (one stream), 1 = two scenes (two streams, crossfaded)
	~ccTyp = 0.6;           // scenes: only the most typical segments of each scene are used (fraction)
	~ccRs = [nil, nil];
	~ccRecentS = [[], []];
	~ccGain = Bus.control(s, 2);
	~ccGain.setn([1, 1]);
	~ccCur = nil;
	~ccSegs = [];
	~ccOrder = [];
	~ccRank = [];

	~ccAudioFiles = {
		var isAudio = { |p| var q = p.toLower; [".wav", ".flac", ".aif", ".aiff"].any { |e| q.endsWith(e) } };
		var raw = (~ccAudioDir ++ "*").pathMatch;
		var list = raw.select(isAudio);
		if(list.isEmpty) {   // second way: ask the shell
			list = ("ls -1 " ++ ~ccQ.(~ccAudioDir)).unixCmdGetStdOut.split($\n)
				.reject { |l| l.isEmpty }.collect { |l| ~ccAudioDir ++ l }.select(isAudio);
		};
		"recordings found in %: % (the folder search matched % entries)".format(~ccAudioDir, list.size, raw.size).postln;
		list.sort
	};
	~ccFileList = ~ccAudioFiles.value;
	~ccTable = { |lenStr| ~ccFile.dirname +/+ (~ccFile.basename.splitext[0] ++ "_seg" ++ lenStr ++ ".csv") };

	// rank of each value between 0 and 1; equal values share the mean rank
	~ccRanks = { |vals|
		var n = vals.size, ord = vals.order, rk = Array.fill(n, 0.0), i = 0, j;
		while { i < n } {
			j = i;
			while { (j + 1 < n) and: { vals[ord[j + 1]] == vals[ord[i]] } } { j = j + 1 };
			(i..j).do { |k| rk.put(ord[k], ((i + j) / 2) / (n - 1).max(1)) };
			i = j + 1;
		};
		rk
	};

	// order the segments by the weighted sum of the ranks of the chosen features
	~ccRecompute = {
		var n = ~ccSegs.size;
		var score = Array.fill(n, 0.0);
		var ends, lowIdx, highIdx, mean, txt;
		~ccFeatures.do { |f, k|
			if(~ccW[k] != 0) { score = score + (~ccNormRank[k] * ~ccW[k]) };
		};
		~ccOrder = score.order;
		if(~ccRev) { ~ccOrder = ~ccOrder.reverse };
		~ccRank = Array.fill(n, 0.0);
		~ccOrder.do { |idx, k| ~ccRank.put(idx, k / (n - 1).max(1)) };
		~ccRecent = [];
		// how different are the two ends? lowest 10 % and highest 10 % of the segments
		ends = max(1, (n / 10).asInteger);
		lowIdx = ~ccOrder[0..(ends - 1)];
		highIdx = ~ccOrder[(n - ends)..(n - 1)];
		mean = { |idx, col| idx.collect { |i| ~ccSegs[i][col] }.mean };
		txt = if(~ccW.abs.sum == 0) { "set at least one weight" } {
			"ends of the scale (lowest 10 % | highest 10 %): loudness % | % dB, hiss % | % dB, activity % | % dB, events % | % per s".format(
				mean.(lowIdx, 2).round(0.1), mean.(highIdx, 2).round(0.1),
				mean.(lowIdx, 8).round(0.1), mean.(highIdx, 8).round(0.1),
				mean.(lowIdx, 6).round(0.1), mean.(highIdx, 6).round(0.1),
				mean.(lowIdx, 9).round(0.1), mean.(highIdx, 9).round(0.1))
		};
		{
			if(~ccEndsLabel.notNil) { ~ccEndsLabel.string = txt };
			if(~ccUv.notNil) { ~ccUv.refresh };
		}.defer;
	};

	// gain of the two streams: ordered mode uses stream 0 at full level; scene mode crossfades (equal power)
	~ccApplyGain = {
		if(~ccMode == 1) {
			~ccGain.setn([(~ccVal * 0.5pi).cos, (~ccVal * 0.5pi).sin])
		} {
			~ccGain.setn([1, 1])
		};
	};

	// scene mode: next segment of scene k (a typical one, not played just before)
	~ccPickScene = { |k|
		var all = (0..(~ccSegs.size - 1)).select { |i| (~ccSegs[i][12] == k) and: { ~ccSegs[i][13] <= ~ccTyp } };
		var dur = ~ccSegs[0][1];
		var cands, idx;
		if(all.isEmpty) { all = (0..(~ccSegs.size - 1)).select { |i| ~ccSegs[i][12] == k } };
		cands = all.reject { |i| ~ccRecentS[k].any { |j| (~ccSegs[i][0] - ~ccSegs[j][0]).abs < dur } };
		if(cands.isEmpty) { cands = all };
		idx = cands.choose;
		~ccRecentS[k] = (~ccRecentS[k] ++ [idx]).keep(-6);
		idx
	};

	// choose the next segment for the current value
	~ccPick = {
		var n = ~ccSegs.size;
		var p = (~ccVal * (n - 1)).round.asInteger;
		var wd = max(1, (~ccSpread * n).round.asInteger);
		var lo = (p - wd).max(0);
		var hi = (p + wd).min(n - 1);
		var all = ~ccOrder[lo..hi];
		var dur = ~ccSegs[0][1];
		var cands = all.reject { |i|
			~ccRecent.any { |j| (~ccSegs[i][0] - ~ccSegs[j][0]).abs < dur }
		};
		var idx;
		if(cands.isEmpty) { cands = all };
		idx = cands.choose;
		~ccRecent = (~ccRecent ++ [idx]).keep(-6);
		idx
	};

	~ccStop = {
		~ccPlaying = false;
		if(~ccR.notNil) { ~ccR.stop; ~ccR = nil };
		~ccRs.do { |r, k| if(r.notNil) { r.stop }; ~ccRs[k] = nil };
	};
	~ccStart = {
		~ccStop.value;
		if(~ccSegs.size > 1 and: { ~ccBuf.notNil }) {
			~ccPlaying = true;
			~ccApplyGain.value;
			if(~ccMode == 1) {
				2.do { |k|
					~ccRs[k] = Routine({
						while { ~ccPlaying and: { ~ccMode == 1 } } {
							var i = ~ccPickScene.(k);
							var seg = ~ccSegs[i];
							var dur = seg[1];
							var xf = min(~ccXf, dur / 2);
							var g = if(k == 0) { (~ccVal * 0.5pi).cos } { (~ccVal * 0.5pi).sin };
							if(g > 0.02) {   // a scene that is (almost) silent at this value is not started
								Synth(if(~ccCh == 1) { \ccSeg1 } { \ccSeg2 },
									[\buf, ~ccBuf, \start, seg[0] * ~ccSR, \dur, dur, \xf, xf, \amp, ~ccAmp,
									\gbus, ~ccGain, \gidx, k]);
							};
							~ccCur = i;
							{ ~ccShow.value }.defer;
							(dur - xf).wait;
						};
					}).play(SystemClock);
				};
			} {
			~ccR = Routine({
				while { ~ccPlaying } {
					var i, seg, dur, xf;
					i = ~ccPick.value;
					seg = ~ccSegs[i];
					dur = seg[1];
					xf = min(~ccXf, dur / 2);
					Synth(if(~ccCh == 1) { \ccSeg1 } { \ccSeg2 },
						[\buf, ~ccBuf, \start, seg[0] * ~ccSR, \dur, dur, \xf, xf, \amp, ~ccAmp,
						\gbus, ~ccGain, \gidx, 0]);
					~ccCur = i;
					{ ~ccShow.value }.defer;
					(dur - xf).wait;
				};
			}).play(SystemClock);
			};
		};
	};

	// load the table for the chosen segment length. No table (or an old one): run make_corpus.py first.
	~ccLoad = {
		var path = ~ccTable.(~ccLenStr.(~ccLenCs));
		var rows;
		if(File.exists(path).not) {
			~ccAnalyse.value;
		} {
			rows = CSVFileReader.read(path, true, true).drop(1).collect { |r| r.collect { |x| x.asFloat } };
			if(rows.isEmpty or: { rows[0].size < 14 }) {
				~ccAnalyse.value;
			} {
				~ccTried = nil;
				~ccSegs = rows;
				~ccRecent = [];
				~ccCur = nil;
				~ccNormRank = ~ccFeatures.collect { |f| ~ccRanks.(rows.collect { |r| r[f[1]] }) };
				~ccRecompute.value;
				{ ~ccStatus.string = "% segments from %".format(rows.size, ~ccFile.basename); ~ccUv.refresh }.defer;
				~ccStart.value;
			};
		};
	};

	~ccAnalyse = {
		var key = ~ccFile ++ ~ccLenCs.asString;
		var cmd = "% % % --lengths %".format(~ccQ.(~ccPy), ~ccQ.(~ccScript), ~ccQ.(~ccFile), ~ccLenStr.(~ccLenCs));
		{ ~ccStatus.string = "analysing % in segments of % s ...".format(~ccFile.basename, ~ccLenStr.(~ccLenCs)) }.defer;
		"running: %".format(cmd).postln;
		cmd.unixCmd({ |code|
			if(code == 0 and: { ~ccTried != key }) {
				~ccTried = key;   // if the table is still unusable, do not analyse the same file again and again
				~ccLoad.value;
			} {
				{ ~ccStatus.string = "analysis failed or gave no usable table (exit code %), see the post window".format(code) }.defer;
			};
		});
	};

	// open a recording: read it into a buffer, then its table
	~ccOpenFile = { |path|
		{
			var sf = SoundFile.new;
			~ccStop.value;
			~ccSegs = [];
			~ccCur = nil;
			if(sf.openRead(path).not) {
				{ ~ccStatus.string = "could not open " ++ path.basename }.defer;
			} {
				~ccFile = path;
				~ccSR = sf.sampleRate;
				~ccCh = sf.numChannels;
				~ccFileDur = sf.numFrames / sf.sampleRate;
				sf.close;
				if(~ccBuf.notNil) { ~ccBuf.free };
				~ccBuf = Buffer.read(s, path);
				s.sync;
				~ccLoad.value;
			};
		}.fork;
	};

	// set the value from code or from the slider
	~ccSet = { |v|
		~ccVal = v.clip(0, 1);
		~ccApplyGain.value;
		{
			if(~ccValSl.notNil) { ~ccValSl.value = ~ccVal };
			if(~ccValLabel.notNil) { ~ccValLabel.string = "value " ++ ~ccVal.round(0.01).asString };
			if(~ccUv.notNil) { ~ccUv.refresh };
		}.defer;
	};

	~ccSweep = { |secs|
		if(~ccSweepR.notNil) { ~ccSweepR.stop; ~ccSweepR = nil };
		if(secs.notNil) {
			~ccSweepR = Routine({
				var steps = (secs / 0.05).asInteger.max(2);
				(2 * steps).do { |k|
					var v = if(k < steps) { k / (steps - 1) } { 1 - ((k - steps) / (steps - 1)) };
					~ccSet.(v);
					0.05.wait;
				};
				{ ~ccSweepBtn.value = 0 }.defer;
			}).play(SystemClock);
		};
	};

	~ccShow = {
		var r;
		if(~ccCur.notNil and: { ~ccLabel.notNil }) {
			r = ~ccSegs[~ccCur];
			~ccLabel.string = "playing %-% s | loudness % dB | hiss % dB | hiss brightness % Hz | rumble % | activity % dB | events %/s".format(
				r[0].round(0.01), (r[0] + r[1]).round(0.01), r[2].round(0.1), r[8].round(0.1), r[7].round(0),
				r[5].round(0.01), r[6].round(0.1), r[9].round(0.1));
		};
		if(~ccUv.notNil) { ~ccUv.refresh };
	};

	// ---- window (built on the AppClock, as all GUI calls must be)
	~ccBuildWindow = {
	if(~ccWin.notNil and: { ~ccWin.isClosed.not }) { ~ccWin.close };

	~ccUv = UserView().minSize_(Size(620, 300)).background_(Color.white);
	~ccUv.drawFunc = { |v|
		var w = v.bounds.width, h = v.bounds.height, l = 36, r = w - 8, top = 8, bot = h - 22;
		var n = ~ccSegs.size;
		var p, wd, lo, hi, xOf, yOf;
		if(n > 1 and: { ~ccFileDur.notNil }) {
			p = (~ccVal * (n - 1)).round.asInteger;
			wd = max(1, (~ccSpread * n).round.asInteger);
			lo = (p - wd).max(0);
			hi = (p + wd).min(n - 1);
			xOf = { |i| l + ((~ccSegs[i][0] + (~ccSegs[i][1] / 2)) / ~ccFileDur * (r - l)) };
			yOf = { |i| bot - (~ccRank[i] * (bot - top)) };
			Pen.fillColor = Color.gray(0.97);
			Pen.fillRect(Rect(l, top, r - l, bot - top));
			Pen.fillColor = Color(0.6, 0.75, 1, 0.25);
			Pen.fillRect(Rect(l, bot - (hi / (n - 1) * (bot - top)), r - l, ((hi - lo) / (n - 1) * (bot - top)).max(2)));
			Pen.fillColor = Color.gray(0.65);
			n.do { |i| Pen.addOval(Rect(xOf.(i) - 2, yOf.(i) - 2, 4, 4)) };
			Pen.fill;
			Pen.fillColor = Color(0.1, 0.4, 0.9);
			(~ccOrder[lo..hi]).do { |i| Pen.addOval(Rect(xOf.(i) - 2.5, yOf.(i) - 2.5, 5, 5)) };
			Pen.fill;
			Pen.fillColor = Color(1, 0.5, 0);
			~ccRecent.do { |i| Pen.addOval(Rect(xOf.(i) - 3.5, yOf.(i) - 3.5, 7, 7)) };
			Pen.fill;
			if(~ccCur.notNil) {
				Pen.strokeColor = Color.black;
				Pen.width = 2;
				Pen.strokeOval(Rect(xOf.(~ccCur) - 7, yOf.(~ccCur) - 7, 14, 14));
			};
			Pen.strokeColor = Color.black;
			Pen.width = 1;
			Pen.line(Point(l, bot - (~ccVal * (bot - top))), Point(r, bot - (~ccVal * (bot - top))));
			Pen.stroke;
			Pen.stringAtPoint("0 s", Point(l, bot + 4), Font("Helvetica", 10), Color.gray(0.3));
			Pen.stringAtPoint(~ccFileDur.round(1).asString ++ " s (position in the recording)",
				Point(r - 190, bot + 4), Font("Helvetica", 10), Color.gray(0.3));
			Pen.stringAtPoint("1", Point(4, top - 2), Font("Helvetica", 10), Color.gray(0.3));
			Pen.stringAtPoint("0", Point(4, bot - 10), Font("Helvetica", 10), Color.gray(0.3));
		};
	};

	~ccLabel = StaticText().string_("");
	~ccStatus = StaticText().string_("");
	~ccValLabel = StaticText().string_("value 0.50");
	~ccValSl = Slider().orientation_(\horizontal).value_(0.5).minSize_(Size(100, 30))
		.action_({ |sl| ~ccSet.(sl.value) });

	~ccFilePop = PopUpMenu().items_(~ccFileList.collect { |f| f.basename }).action_({ |m|
		if(~ccFileList.size > 0) { ~ccOpenFile.(~ccFileList[m.value]) };
	});
	~ccRescanBtn = Button().states_([["rescan folder"]]).action_({
		var cur = ~ccFile;
		~ccFileList = ~ccAudioFiles.value;
		if(cur.notNil and: { ~ccFileList.includesEqual(cur).not }) { ~ccFileList = ~ccFileList ++ [cur] };
		~ccFilePop.items = ~ccFileList.collect { |f| f.basename };
		~ccFilePop.value = ~ccFileList.indexOfEqual(cur) ? 0;
	});
	~ccChooseBtn = Button().states_([["choose another file..."]]).action_({
		FileDialog({ |path|
			if(~ccFileList.includesEqual(path).not) { ~ccFileList = ~ccFileList ++ [path] };
			~ccFilePop.items = ~ccFileList.collect { |f| f.basename };
			~ccFilePop.value = ~ccFileList.indexOfEqual(path);
			~ccOpenFile.(path);
		}, {}, 1, 0, true);
	});
	~ccAnalyseBtn = Button().states_([["analyse again"]]).action_({
		if(~ccFile.notNil) { ~ccTried = nil; ~ccAnalyse.value };
	});

	// one weight per feature: the segments are ordered by the weighted sum of their ranks
	~ccWLabels = ~ccFeatures.collect { |f, k| StaticText().string_(f[0] ++ ": " ++ ~ccW[k].asString) };
	~ccWSl = ~ccFeatures.collect { |f, k|
		Slider().orientation_(\horizontal).value_(~ccW[k].linlin(-1, 1, 0, 1)).action_({ |sl|
			var w = sl.value.linlin(0, 1, -1, 1).round(0.05);
			if(w.abs < 0.1) { w = 0 };
			~ccW[k] = w;
			~ccWLabels[k].string = f[0] ++ ": " ++ w.asString;
			~ccRecompute.value;
		})
	};
	// order of the weights: loudness, hiss amount, hiss brightness, rumble, activity, noisiness, overall brightness, events, timbre 1, timbre 2
	~ccPresets = [
		["timbre axis 1 only (MFCC)", [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]],
		["timbre axis 2 only (MFCC)", [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]],
		["wash to crash: steady and hissy to bursty and rumbly (Seawaves1)", [0, -1, 0, 1, 1, 0, 0, 1, 0, 0]],
		["bursty only: activity and event density", [0, 0, 0, 0, 1, 0, 0, 1, 0, 0]],
		["hiss amount only", [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]],
		["loudness only", [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
	];
	~ccPresetPop = PopUpMenu().items_(~ccPresets.collect { |p| p[0] }).value_(2).action_({ |m|
		~ccW = ~ccPresets[m.value][1].copy;
		~ccFeatures.do { |f, k|
			~ccWSl[k].value = ~ccW[k].linlin(-1, 1, 0, 1);
			~ccWLabels[k].string = f[0] ++ ": " ++ ~ccW[k].asString;
		};
		~ccRecompute.value;
	});
	~ccEndsLabel = StaticText().string_("");
	~ccSceneInfo = {
		var st;
		if(~ccSegs.isEmpty) { "" } {
			st = { |k|
				var t = ~ccSegs.select { |r| r[12] == k }.collect { |r| r[0] }.sort;
				"% segments, from %-% s of the file (central 90 percent)".format(t.size, t[(t.size * 0.05).asInteger].round(1), t[(t.size * 0.95).asInteger.min(t.size - 1)].round(1))
			};
			"scene at value 0: %.  scene at value 1: %.".format(st.(0), st.(1))
		}
	};
	~ccModeBtn = Button().states_([["mode: ordered by features"], ["mode: two scenes, crossfaded"]])
		.action_({ |b|
			var wasPlaying = ~ccPlaying;
			~ccStop.value;
			~ccMode = b.value;
			if(~ccMode == 1) { ~ccEndsLabel.string = ~ccSceneInfo.value } { ~ccRecompute.value };
			~ccApplyGain.value;
			if(wasPlaying) { ~ccStart.value };
		});
	~ccTypLabel = StaticText().string_("scene mode: most typical 60 % of each scene");
	~ccTypSl = Slider().orientation_(\horizontal).value_((0.6 - 0.1) / 0.9)
		.action_({ |sl|
			~ccTyp = 0.1 + (sl.value * 0.9);
			~ccTypLabel.string = "scene mode: most typical " ++ (~ccTyp * 100).round(1).asInteger.asString ++ " % of each scene";
		});
	~ccRevBtn = Button().states_([["reverse: off"], ["reverse: on"]])
		.action_({ |b| ~ccRev = (b.value == 1); ~ccRecompute.value });
	~ccLenLabel = StaticText().string_("segments of 1 s");
	~ccLenPop = Slider().orientation_(\horizontal).value_(~ccLenCs.linlin(10, 100, 0, 1))
		.action_({ |sl|
			var tok;
			~ccLenCs = (sl.value.linlin(0, 1, 10, 100) / 5).round.asInteger * 5;
			~ccLenLabel.string = "segments of % s".format(~ccLenStr.(~ccLenCs));
			~ccLenToken = ~ccLenToken + 1;
			tok = ~ccLenToken;
			// load (or, for a length not analysed yet, analyse) once the slider has rested for 0.5 s
			AppClock.sched(0.5, { if(tok == ~ccLenToken and: { ~ccFile.notNil }) { ~ccLoad.value }; nil });
		});

	~ccSpreadLabel = StaticText().string_("spread: 5 % of the segments");
	~ccSpreadSl = Slider().orientation_(\horizontal).value_(0.05 / 0.3)
		.action_({ |sl|
			~ccSpread = (sl.value * 0.3).max(0.005);
			~ccSpreadLabel.string = "spread: " ++ (~ccSpread * 100).round(0.5).asString ++ " % of the segments";
			~ccUv.refresh;
		});
	~ccXfSpec = ControlSpec(0.02, 1.0, \exp);
	~ccXfLabel = StaticText().string_("crossfade: 300 ms");
	~ccXfSl = Slider().orientation_(\horizontal).value_(~ccXfSpec.unmap(0.3))
		.action_({ |sl|
			~ccXf = ~ccXfSpec.map(sl.value);
			~ccXfLabel.string = "crossfade: % ms".format((~ccXf * 1000).round(1));
		});
	~ccAmpSpec = ControlSpec(0.05, 3, \exp);
	~ccAmpSl = Slider().orientation_(\horizontal).value_(~ccAmpSpec.unmap(0.7))
		.action_({ |sl| ~ccAmp = ~ccAmpSpec.map(sl.value) });

	~ccSweepBox = NumberBox().value_(20).clipLo_(2).clipHi_(120).decimals_(0);
	~ccSweepBtn = Button().states_([["sweep 0-1-0"], ["stop sweep"]])
		.action_({ |b| if(b.value == 1) { ~ccSweep.(~ccSweepBox.value) } { ~ccSweep.(nil) } });
	~ccPlayBtn = Button().states_([["stop"], ["play"]])
		.action_({ |b| if(b.value == 0) { ~ccStart.value } { ~ccStop.value } });

	~ccWin = Window("Concatenative synthesis test", Rect(20, 20, 1100, 900));
	~ccWin.layout = HLayout(
		[VLayout([~ccUv, stretch: 1], HLayout(~ccValSl, ~ccValLabel), ~ccLabel, ~ccEndsLabel, ~ccStatus), stretch: 3],
		[VLayout(
			StaticText().string_("recording"), ~ccFilePop, HLayout(~ccRescanBtn, ~ccChooseBtn), ~ccAnalyseBtn,
			StaticText().string_("the value controls the rank of a combination of features"), ~ccPresetPop,
			*(~ccFeatures.collect { |f, k| [~ccWLabels[k], ~ccWSl[k]] }.flatten ++ [
			~ccModeBtn, ~ccTypLabel, ~ccTypSl,
			~ccRevBtn,
			~ccLenLabel, ~ccLenPop,
			~ccSpreadLabel, ~ccSpreadSl,
			~ccXfLabel, ~ccXfSl,
			StaticText().string_("volume"), ~ccAmpSl,
			HLayout(~ccSweepBtn, StaticText().string_("seconds"), ~ccSweepBox),
			~ccPlayBtn,
			nil
			])
		), stretch: 1]
	);
	~ccWin.onClose = { ~ccStop.value; ~ccSweep.(nil) };
	~ccWin.front;

	// open the first recording: Seawaves1.wav if it is there, otherwise the first one in the folder
	if(~ccFileList.isEmpty) {
		~ccStatus.string = "no recordings found in " ++ ~ccAudioDir;
	} {
		~ccFilePop.value = ~ccFileList.detectIndex { |f| f.basename == "Seawaves1.wav" } ? 0;
		~ccOpenFile.(~ccFileList[~ccFilePop.value]);
	};
	};
	{ ~ccBuildWindow.value }.defer;
};
)

// Stop everything
// ~ccStop.value; ~ccSweep.(nil); ~ccBuf.free;
