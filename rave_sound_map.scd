/*
Sound map: a 2D map of audio clips. Move the mouse over a dot to hear its clip.

Made with latent_map.py. Every dot is one short clip with its embedding (latent vector). The dots are grouped
into clusters (k-means on the latent vectors; latent_map.py chooses the number, which is unrelated to the
number of latent dimensions of the model). No RAVE model and no nn.ar are needed here: this plays saved clips.

- Map menu: lists every map in the latent_maps folder (one per model, or per set of recordings).
- Hover: plays the clip under the mouse (switch off with "play on hover"). Click or drag also plays.
- Clusters: switch clusters on or off in the right panel. Only the "on" clusters can be played, and the rest
  are greyed out. "exclusive select" makes a click on a cluster switch all the others off.
- Views: islands (clusters shown as separate groups, a layout made for display), t-SNE, PCA.
  In t-SNE and PCA the clusters overlap, because the latent space is continuous.
- The text under the map gives the source of the clip, its cluster, brightness (spectral centroid) and level.
- Running block 1 is enough: the window opens by itself. Block 2 only reopens it.
- "Make a new map": runs latent_map.py for a model (and optionally a folder of your recordings).
  This needs a Python with these packages: torch numpy scikit-learn matplotlib plotly soundfile scipy.
  In Terminal, in the RAVE_SC folder:
      .venv/bin/pip install torch numpy scikit-learn matplotlib plotly soundfile scipy
  Use "check python" in the window to test it. Making a map takes about 1 minute.

I could not run SuperCollider, so this is untested. If something fails, send me the post window text.
*/

// 1. Setup, first map and window (server options are taken from your startup file: 48000 Hz, 2 in, 2 out)
//    Run once or again: put the cursor inside the brackets and press Cmd+Enter
(
s.waitForBoot {
	~baseDir   = "/Users/marcellolussana/Documents/Software/RAVE_SC/";
	~mapDir    = ~baseDir ++ "latent_maps/";
	~modelsDir = ~baseDir ++ "models/";
	~pyBin     = ~baseDir ++ ".venv/bin/python";   // a Python that has torch etc. (see above)
	~pyScript  = ~baseDir ++ "latent_map.py";

	SynthDef(\clip, { |buf, start = 0, dur = 0.3, amp = 0.5, gate = 1, out = 0,
		lpf = 20000, atk = 0.03, rel = 0.03|
		var live = Line.kr(1, 0, (dur - rel).max(0.01)) > 0;   // release starts rel before the end of the clip
		var env = EnvGen.kr(Env.asr(atk, 1, rel), gate * live, doneAction: 2);
		var sig = PlayBuf.ar(1, buf, BufRateScale.kr(buf), startPos: start, loop: 0);
		sig = LPF.ar(LPF.ar(sig, lpf), lpf);   // two stages: 24 dB per octave
		Out.ar(out, Pan2.ar(sig * env, 0, amp));
	}).add;
	s.sync;

	~amp = 0.5;
	~lenFrac = 1.0;   // how much of each clip is played (0.1 .. 1)
	~clipDur = 0.3;     // length of one stored clip in seconds (replaced when a map is loaded)
	~lpf = 20000;     // low-pass cutoff in Hz
	~fade = 0.03;     // fade in and fade out in seconds
	~hover = true;
	~exclusive = false;
	~loading = false;
	~viewIdx = 0;
	~lastIdx = nil;
	~lastT = 0;
	~last = nil;
	~buf = nil;
	~cur = nil;
	~xs = [0]; ~ys = [0]; ~mask = [0]; ~centers = [];
	~byGroup = Dictionary.new;
	~groupIds = [];
	~active = [];

	~findMaps = {
		PathName(~mapDir).files
			.select { |f| f.fileName.endsWith("_map.csv") }
			.collect { |f| f.fileName.replace("_map.csv", "") }
			.sort
	};
	~maps = ~findMaps.value;

	~updateMask = { ~mask = ~grp.collect { |g| if(~active[g]) { 0 } { 1e6 } } };

	~palFor = { |n|
		(if(n <= 10) {
			["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
				"#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
		} {
			["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a", "#d62728", "#ff9896",
				"#9467bd", "#c5b0d5", "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7",
				"#bcbd22", "#dbdb8d", "#17becf", "#9edae5"]
		}).collect { |h| Color.fromHexString(h) }
	};
	~pal = ~palFor.(8);
	~colorOf = { |g| ~pal[g % ~pal.size] };

	// view 0 = islands, 1 = t-SNE, 2 = PCA. Positions are scaled to 0..1 (islands keep their aspect ratio).
	~setView = { |i|
		var c = i * 2;
		var xs = ~rows.collect { |r| r[c] };
		var ys = ~rows.collect { |r| r[c + 1] };
		var xr = xs.maxItem - xs.minItem;
		var yr = ys.maxItem - ys.minItem;
		var xmid = (xs.maxItem + xs.minItem) / 2;
		var ymid = (ys.maxItem + ys.minItem) / 2;
		var xsc = if(i == 0) { max(xr, yr) } { xr };
		var ysc = if(i == 0) { max(xr, yr) } { yr };
		~viewIdx = i;
		~xs = (xs - xmid) / xsc * 0.94 + 0.5;
		~ys = (ys - ymid) / ysc * 0.94 + 0.5;
		~centers = ~groupIds.collect { |g|
			var ids = ~byGroup[g];
			[g, ids.collect { |k| ~xs[k] }.mean, ids.collect { |k| ~ys[k] }.mean]
		};
		~cur = nil;
		if(~uv.notNil) { { ~uv.refresh }.defer };   // GUI calls must run on the AppClock
	};

	// load one map: CSV, clusters, and the audio file with all clips
	~loadMap = { |name|
		{
			var base = ~mapDir ++ name;
			var buf;
			~loading = true;
			~curMap = name;
			if(~last.notNil and: { ~last.isPlaying }) { ~last.set(\gate, 0) };

			// CSV columns: x,y islands | x,y t-SNE | x,y PCA | cluster | centroid Hz | level dB | latent vector
			~rows = CSVFileReader.read(base ++ "_map.csv", true, true)
				.collect { |r| r.collect { |x| x.asFloat } };
			~grp  = ~rows.collect { |r| r[6].asInteger };
			~cen  = ~rows.collect { |r| r[7] };
			~lvl  = ~rows.collect { |r| r[8] };
			~info = File.readAllString(base ++ "_map_info.txt").split($\n);
			~gnames = Dictionary.new;
			~meta = Dictionary.new;
			File.readAllString(base ++ "_map_groups.txt").split($\n).do { |l|
				var p = l.split($|);
				if(p.size == 2) {
					if(p[0].beginsWith("#")) { ~meta[p[0]] = p[1] } { ~gnames[p[0].asInteger] = p[1] };
				};
			};
			~clipLen = ~meta["#clip_samples"].asInteger;
			~clipDur = ~clipLen / ~meta["#sr"].asFloat;

			buf = Buffer.read(s, ~mapDir ++ ~meta["#audio"]);
			s.sync;
			if(~buf.notNil) { ~buf.free };
			~buf = buf;
			if(~sf.notNil) { ~sf.close };   // the same file, read from disk for the spectrogram of one clip at a time
			~sf = SoundFile.new;
			if(~sf.openRead(~mapDir ++ ~meta["#audio"]).not) { "could not open the clips file for the spectrogram".postln; ~sf = nil };
			"% of % clips are almost silent (rms below -70 dB)".format(~lvl.select { |x| x < -70 }.size, ~lvl.size).postln;

			~byGroup = Dictionary.new;
			~grp.do { |g, i| ~byGroup[g] = (~byGroup[g] ? []).add(i) };
			~groupIds = ~byGroup.keys.asArray.sort;
			~active = Array.fill(~groupIds.size, true);
			~pal = ~palFor.(~groupIds.size);
			~updateMask.value;
			~setView.(~viewIdx);
			~lastIdx = nil;
			~loading = false;
			{ ~refreshGui.value }.defer;
			"sound map ready: %  (% clips, % clusters)".format(name, ~rows.size, ~groupIds.size).postln;
		}.fork;
	};

	~play = { |i|
		var d = ~clipDur * ~lenFrac;
		var f = min(~fade, d / 2);
		if(~last.notNil and: { ~last.isPlaying }) { ~last.set(\gate, 0) };
		~last = Synth(\clip, [\buf, ~buf, \start, i * ~clipLen, \dur, d, \amp, ~amp,
			\lpf, ~lpf, \atk, f, \rel, f]).register;
	};

	// ---- spectrogram of one clip (read from the clips file, FFT 1024, Hann window)
	~specPal = {
		var stops = [Color(0, 0, 0.02), Color(0.2, 0.05, 0.35), Color(0.55, 0.1, 0.45),
			Color(0.9, 0.35, 0.3), Color(1, 0.75, 0.3), Color(1, 0.98, 0.75)];
		16.collect { |q|
			var t = q / 15 * 5;
			var k = t.floor.min(4);
			stops[k].blend(stops[k + 1], t - k)
		}
	}.value;
	~spec = nil;
	~specRows = 64;
	~specCalc = { |i|
		var n = 1024, per = 8;   // 512 bins in 64 rows of 8 bins (the loudest bin of each group is shown)
		var hop, data, nCols, cols, frame, z, mag, ref, peakDb, rmsDb;
		hop = max(128, (~clipLen / 200).ceil.asInteger);
		data = FloatArray.newClear(~clipLen);
		~sf.seek(i * ~clipLen, 0);
		~sf.readData(data);
		nCols = ((~clipLen - n) / hop).floor.asInteger + 1;
		if(~specWin.isNil) { ~specWin = Signal.hanningWindow(n); ~specTab = Signal.fftCosTable(n) };
		cols = nCols.collect { |c|
			frame = Signal.newFrom(data.copyRange(c * hop, c * hop + n - 1)) * ~specWin;
			z = frame.fft(Signal.newClear(n), ~specTab);
			mag = z.magnitude.copyRange(0, (n div: 2) - 1).as(Array);   // n div: 2 is an integer (n / 2 would be 512.0)
			mag.clump(per).collect { |g| g.maxItem }
		};
		cols = cols.collect { |col| (col / (n / 4)).max(1e-6).ampdb };   // dB relative to a full-scale sine
		ref = max(cols.flatten.maxItem, -50);   // colour scale: 70 dB below the loudest bin (at most -50 dB)
		cols = cols.collect { |col| ((col - (ref - 70)) / 70).clip(0, 1) };
		peakDb = data.abs.maxItem.max(1e-6).ampdb;
		rmsDb = data.squared.mean.sqrt.max(1e-6).ampdb;
		[cols, peakDb, rmsDb, ref]
	};
	~showSpec = { |i|
		var r;
		if(~sv.notNil and: { ~sf.notNil }) {
			r = ~specCalc.(i);
			~spec = r[0];
			~specLabel.string = "clip %: peak % dB, rms % dB%   |   colour scale: % dB (bright) to % dB (dark)".format(
				i, r[1].round(0.1), r[2].round(0.1), if(r[2] < -70) { "  (almost silent)" } { "" },
				r[3].round(1), (r[3] - 70).round(1));
			~sv.refresh;
		};
	};

	// x, y: pixel position in the map view. trigger: play the clip. force: play even if it is the same dot.
	~at = { |x, y, trigger, force|
		var w, h, sq, mx, my, d2, i, now;
		if(~loading.not and: { ~uv.notNil }) {
			w = ~uv.bounds.width; h = ~uv.bounds.height; sq = min(w, h);
			mx = (x - ((w - sq) / 2)) / sq;
			my = 1 - ((y - ((h - sq) / 2)) / sq);
			d2 = (~xs - mx).squared + (~ys - my).squared + ~mask;
			i = d2.minIndex;
			now = Main.elapsedTime;
			if(d2[i] < 0.0049) {   // the mouse is within 0.07 of a dot
				~cur = i;
				~label.string = "%   |   cluster %   |   brightness % Hz, level % dB".format(
					~info[i], ~grp[i], ~cen[i].round(1), ~lvl[i].round(1));
				if(trigger and: { force or: { (i != ~lastIdx) and: { (now - ~lastT) > 0.04 } } }) {
					~play.(i);
					~showSpec.(i);
					~lastIdx = i;
					~lastT = now;
				};
			} {
				~cur = nil;
				~lastIdx = nil;   // leaving a dot: it can be played again when the mouse returns
			};
			~uv.refresh;
		};
	};

	// ---- make a new map with latent_map.py
	~q = { |p| "\"" ++ p ++ "\"" };
	~audioDir = nil;
	~nClips = 600;
	~checkPython = {
		("% -c \"import torch, sklearn, plotly, matplotlib, soundfile, scipy; print('python packages ok')\"")
			.format(~q.(~pyBin)).unixCmd({ |code|
				if(code != 0) { "python check failed: install the packages (see the top of this file)".postln };
			});
	};
	~makeMap = { |modelFile, audioDir, n, clipSec = 0.3, spread = 1.0, walk = 0.0|
		var suffix = (if(clipSec == 0.3) { "" } { "_" ++ clipSec.asString ++ "s" })
			++ (if(spread != 1.0) { "_s" ++ spread.asString } { "" })
			++ (if(walk > 0) { "_w" ++ walk.asString } { "" });
		var cmd = "% -u % --model % --out % --n % --clip %".format(
			~q.(~pyBin), ~q.(~pyScript), ~q.(~modelsDir ++ modelFile), ~q.(~mapDir), n, clipSec);
		var newName = modelFile.splitext[0] ++ (if(audioDir.notNil) { "_audio" } { "_decoder" }) ++ suffix;
		if(audioDir.notNil) { cmd = cmd ++ " --audio " ++ ~q.(audioDir) };
		if(suffix != "") { cmd = cmd ++ " --suffix " ++ suffix };
		if(audioDir.isNil) { cmd = cmd ++ " --scale " ++ spread ++ " --walk " ++ walk };
		"running: %".format(cmd).postln;
		if(~status.notNil) {
			~status.string = "making %: % clips of % s = % s of audio (about % MB in memory). Progress is in the post window."
				.format(newName, n, clipSec, (n * clipSec).round(1), (n * clipSec * 48000 * 4 / 1e6).round(1));
		};
		cmd.unixCmd({ |code|
			{
				if(code == 0 and: { File.exists(~mapDir ++ newName ++ "_map.csv") }) {
					~maps = ~findMaps.value;
					if(~mapPop.notNil) { ~mapPop.items = ~maps };
					~status.string = "map made: %".format(newName);
					~loadMap.(newName);
				} {
					~status.string = "latent_map.py failed or made no map (exit code %), see the post window".format(code);
				};
			}.defer;
		});
	};

	// the window: built by a function, so it opens by itself and can be reopened with ~openWindow.value
	~openWindow = {
	if(~win.notNil and: { ~win.isClosed.not }) { ~win.close };

	~uv = UserView().minSize_(Size(560, 560)).background_(Color.white).acceptsMouseOver_(true);
	~uv.drawFunc = { |v|
		var w = v.bounds.width, h = v.bounds.height, sq = min(w, h);
		var ox = (w - sq) / 2, oy = (h - sq) / 2;
		Pen.fillColor = Color.gray(0.97);
		Pen.fillRect(Rect(ox, oy, sq, sq));
		if(~loading.not) {
			~byGroup.keysValuesDo { |g, ids|   // clusters that are off, in grey
				if(~active[g].not) {
					Pen.fillColor = Color.gray(0.8, 0.5);
					ids.do { |i| Pen.addOval(Rect(ox + (~xs[i] * sq) - 2, oy + ((1 - ~ys[i]) * sq) - 2, 4, 4)) };
					Pen.fill;
				};
			};
			~byGroup.keysValuesDo { |g, ids|   // clusters that are on, in colour
				var c = ~colorOf.(g);
				if(~active[g]) {
					Pen.fillColor = Color(c.red, c.green, c.blue, 0.85);
					ids.do { |i| Pen.addOval(Rect(ox + (~xs[i] * sq) - 2.5, oy + ((1 - ~ys[i]) * sq) - 2.5, 5, 5)) };
					Pen.fill;
				};
			};
			~centers.do { |t|
				if(~active[t[0]]) {
					Pen.stringAtPoint(t[0].asString,
						Point(ox + (t[1] * sq) - 5, oy + ((1 - t[2]) * sq) - 11),
						Font("Helvetica", 18, true), Color.black);
				};
			};
			if(~cur.notNil) {
				Pen.strokeColor = Color.black;
				Pen.width = 2;
				Pen.strokeOval(Rect(ox + (~xs[~cur] * sq) - 8, oy + ((1 - ~ys[~cur]) * sq) - 8, 16, 16));
			};
		};
	};
	~uv.mouseOverAction = { |v, x, y| ~at.(x, y, ~hover, false) };
	~uv.mouseMoveAction = { |v, x, y| ~at.(x, y, true, false) };   // dragging plays too
	~uv.mouseDownAction = { |v, x, y| ~at.(x, y, true, true) };    // a click always plays

	~label = StaticText().string_("move the mouse over the dots");

	~sv = UserView().minSize_(Size(400, 150)).background_(Color.black);
	~sv.drawFunc = { |v|
		var w = v.bounds.width, h = v.bounds.height, l = 44, top = 6, pw = w - l - 6, ph = h - top - 18;
		var rows = ~specRows, nCols, cw, ch, buckets, nyq, y;
		Pen.fillColor = ~specPal[0];
		Pen.fillRect(Rect(l, top, pw, ph));
		if(~spec.notNil) {
			nCols = ~spec.size;
			cw = pw / nCols;
			ch = ph / rows;
			buckets = Array.fill(16) { List.new };
			~spec.do { |col, c|
				col.do { |val, r|
					var q = (val * 15).round.asInteger;
					if(q > 0) { buckets[q].add(Rect(l + (c * cw), top + ph - ((r + 1) * ch), cw + 0.6, ch + 0.6)) };
				};
			};
			buckets.do { |bk, q|
				if(bk.size > 0) {
					Pen.fillColor = ~specPal[q];
					bk.do { |rc| Pen.addRect(rc) };
					Pen.fill;
				};
			};
		};
		nyq = (if(~meta.notNil) { (~meta["#sr"] ? "48000").asFloat } { 48000.0 }) / 2;
		[0, 5000, 10000, 15000, 20000].do { |f|
			if(f <= nyq) {
				y = top + ph - (f / nyq * ph);
				Pen.stringAtPoint((f / 1000).asInteger.asString ++ " kHz", Point(2, (y - 12).max(0)), Font("Helvetica", 10), Color.gray(0.8));
			};
		};
		Pen.stringAtPoint("0 s", Point(l, h - 15), Font("Helvetica", 10), Color.gray(0.8));
		Pen.stringAtPoint(~clipDur.round(0.01).asString ++ " s", Point(l + pw - 34, h - 15), Font("Helvetica", 10), Color.gray(0.8));
	};
	~specLabel = StaticText().string_("spectrogram of the last clip played");

	// map (model) selection
	~mapPop = PopUpMenu().items_(~maps).action_({ |m| ~loadMap.(~maps[m.value]) });
	~rescanBtn = Button().states_([["rescan folder"]]).action_({
		~maps = ~findMaps.value;
		~mapPop.items = ~maps;
		~mapPop.value = ~maps.indexOfEqual(~curMap) ? 0;
	});

	~viewPop = PopUpMenu().items_(["islands (grouped)", "t-SNE", "PCA"]).value_(0)
		.action_({ |m| ~setView.(m.value) });
	~hoverBtn = Button().states_([["play on hover: off"], ["play on hover: on"]]).value_(1)
		.action_({ |b| ~hover = (b.value == 1) });

	// up to 20 cluster buttons, only as many as the map has are shown
	~clusterBtns = Array.fill(20) { |i|
		Button().states_([[i.asString, Color.gray(0.4), Color.gray(0.9)], [i.asString, Color.white, Color.gray(0.5)]])
			.value_(1).visible_(false).action_({ |b|
				if(~exclusive) {
					~active = Array.fill(~groupIds.size, false);
					~active[i] = true;
					~clusterBtns.do { |o, k| o.value = if(k == i) { 1 } { 0 } };
				} {
					~active[i] = (b.value == 1);
				};
				~updateMask.value;
				~uv.refresh;
			})
	};
	~exclBtn = Button().states_([["exclusive select: off"], ["exclusive select: on"]])
		.action_({ |b| ~exclusive = (b.value == 1) });
	~allBtn = Button().states_([["all on"]]).action_({
		~active = Array.fill(~groupIds.size, true);
		~clusterBtns.do { |o, k| if(k < ~groupIds.size) { o.value = 1 } };
		~updateMask.value;
		~uv.refresh;
	});
	~noneBtn = Button().states_([["all off"]]).action_({
		~active = Array.fill(~groupIds.size, false);
		~clusterBtns.do { |o| o.value = 0 };
		~updateMask.value;
		~uv.refresh;
	});

	~ampSpec = ControlSpec(0.1, 4, \exp);
	~ampSl = Slider().orientation_(\horizontal).value_(~ampSpec.unmap(0.5))
		.action_({ |sl| ~amp = ~ampSpec.map(sl.value) });

	// smoothing: low-pass filter and fades
	~lpfSpec = ControlSpec(500, 20000, \exp);
	~fadeSpec = ControlSpec(0.002, 0.15, \exp);
	~lpfLabel = StaticText().string_("low-pass: off");
	~lpfSl = Slider().orientation_(\horizontal).value_(1.0).action_({ |sl|
		~lpf = ~lpfSpec.map(sl.value);
		~lpfLabel.string = if(~lpf > 19000) { "low-pass: off" } { "low-pass: % Hz".format(~lpf.round(10).asInteger) };
	});
	~fadeLabel = StaticText().string_("fade in / out: 30 ms");
	~fadeSl = Slider().orientation_(\horizontal).value_(~fadeSpec.unmap(0.03)).action_({ |sl|
		~fade = ~fadeSpec.map(sl.value);
		~fadeLabel.string = "fade in / out: % ms".format((~fade * 1000).round(1));
	});

	// how much of each stored clip is played
	~lenLabel = StaticText().string_("sample length");
	~lenSl = Slider().orientation_(\horizontal).value_(1.0).action_({ |sl|
		~lenFrac = sl.value.linlin(0, 1, 0.1, 1);
		~lenLabel.string = ~lenText.value;
	});

	// make a new map
	~modelFiles = PathName(~modelsDir).files.select { |f| f.extension == "ts" }.collect { |f| f.fileName }.sort;
	~modelPop = PopUpMenu().items_(~modelFiles);
	~audioLabel = StaticText().string_("recordings: none (decoder map)");
	~audioBtn = Button().states_([["choose recordings folder..."]]).action_({
		FileDialog({ |path|
			~audioDir = path;
			~audioLabel.string = "recordings: " ++ path.basename;
		}, {}, 2, 0, true);
	});
	~noAudioBtn = Button().states_([["no recordings"]]).action_({
		~audioDir = nil;
		~audioLabel.string = "recordings: none (decoder map)";
	});
	~clipChoices = [0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0];
	~clipBox = PopUpMenu().items_(~clipChoices.collect { |c| c.asString ++ " s" }).value_(0);
	~spreadBox = NumberBox().value_(1.0).clipLo_(0.2).clipHi_(2).decimals_(2);
	~walkBox = NumberBox().value_(0.0).clipLo_(0).clipHi_(2).decimals_(2);
	~nBox = NumberBox().value_(600).clipLo_(100).clipHi_(2000).decimals_(0).action_({ |n| ~nClips = n.value.asInteger });
	~makeBtn = Button().states_([["make map"]]).action_({
		~makeMap.(~modelFiles[~modelPop.value], ~audioDir, ~nBox.value.asInteger, ~clipChoices[~clipBox.value],
			~spreadBox.value, ~walkBox.value);
	});
	~checkBtn = Button().states_([["check python"]]).action_({ ~checkPython.value });
	~status = StaticText().string_("");

	~win = Window("RAVE sound map", Rect(40, 40, 1100, 860));
	~win.layout = HLayout(
		[VLayout([~uv, stretch: 4], ~label, [~sv, stretch: 1], ~specLabel), stretch: 3],
		[VLayout(
			StaticText().string_("map (model)"), HLayout(~mapPop, ~rescanBtn),
			StaticText().string_("view"), ~viewPop, ~hoverBtn,
			StaticText().string_("clusters (click to switch on / off)"),
			~exclBtn, HLayout(~allBtn, ~noneBtn),
			*(~clusterBtns ++ [
				StaticText().string_("volume"), ~ampSl,
				StaticText().string_("sample length (plays the start of each clip)"), ~lenSl, ~lenLabel,
				StaticText().string_("smoothing"), ~lpfSl, ~lpfLabel, ~fadeSl, ~fadeLabel,
				StaticText().string_("Make a new map"), ~modelPop, ~audioLabel,
				HLayout(~audioBtn, ~noAudioBtn),
				HLayout(StaticText().string_("clips"), ~nBox, StaticText().string_("seconds each"), ~clipBox),
				HLayout(StaticText().string_("spread (1 = full)"), ~spreadBox, StaticText().string_("latent drift per s"), ~walkBox),
				HLayout(~makeBtn, ~checkBtn),
				~status
			])
		), stretch: 1]
	);
	~win.acceptsMouseOver = true;   // needed, otherwise the map gets no mouse-over events without a click
	~win.front;
	~lenText = { var cd = ~clipDur ? 0.3; "length: % s (of % s stored)".format((cd * ~lenFrac).round(0.01), cd.round(0.01)) };
	~refreshGui = {
		if(~win.notNil and: { ~win.isClosed.not }) {
			~clusterBtns.do { |b, i|
				if(i < ~groupIds.size) {
					b.states = [
						[~gnames[i], Color.gray(0.4), Color.gray(0.9)],
						[~gnames[i], Color.white, ~colorOf.(i)]
					];
					b.value = 1;
					b.visible = true;
				} {
					b.visible = false;
				};
			};
			~mapPop.items = ~maps;
			~mapPop.value = ~maps.indexOfEqual(~curMap) ? 0;
			~lenLabel.string = ~lenText.value;
			~uv.refresh;
		};
	};
	~refreshGui.value;
	};

	if(~maps.isEmpty) {
		"No maps found in %. Run latent_map.py or use 'Make a new map' in the window.".format(~mapDir).postln;
	} {
		~loadMap.(~maps.detect { |m| m.contains("dawnchorus") } ? ~maps[0]);
	};
	{ ~openWindow.value }.defer;   // open the window as soon as everything is loaded
};
)

// 2. Reopen the window (if you closed it)
~openWindow.value;


// Stop all sound and free the clips
// ~last.free; ~buf.free;
