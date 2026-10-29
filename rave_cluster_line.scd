/*
Cluster line: one cluster of a sound map, laid out on a line from 0 to 1.

It uses the maps made by latent_map.py (same folder as rave_sound_map.scd). The clips of ONE cluster are
sorted along a single axis, so a value between 0 and 1 (like the proximity value in Sentire) picks a clip:
0 = one end of the line, 1 = the other end. This is for hearing what a 0-1 control would do inside a cluster.

- Map: choose the map (model). Cluster: choose which cluster is laid out.
- Order by: what the line follows.
    brightness  spectral centroid of the clip
    level       loudness of the clip
    main latent axis   the direction along which the latent vectors of this cluster vary most (PC1). The sign is
                       chosen so that it points towards brighter clips. This is the direction one would move in a
                       live RAVE model. Whether it is "tonal" depends on the model, check the correlations shown.
    latent dim n       one latent dimension
- Spacing: "even" puts the clips at equal distances (value 0.5 = the median clip). "by value" spaces them by the
  size of the key, so the value is linear in brightness, level, or latent value.
- Reverse: swaps the two ends.
- Value: the slider, or the mouse over the line. The vertical line shows the value, the ring the clip it picks.
- Height of the dots: the other quantity (level when ordered by brightness, brightness otherwise). If the dots
  also rise and fall with the line, that other quantity changes together with the key.
- Loop: keeps replaying the current clip, one after the other with a short crossfade, so you hear a continuous
  sound that changes while you move the value. Without it, a clip plays once when the value reaches a new clip.
- Sweep: moves the value up and down automatically. Use it together with Loop.
- Sample length, low-pass and fade work as in rave_sound_map.scd.

Run block 1 (the window opens by itself). Block 2 reopens it.
I could not run SuperCollider, so this is untested. If something fails, send me the post window text.
*/

(
s.waitForBoot {
	~lbaseDir = "/Users/marcellolussana/Documents/Software/RAVE_SC/";
	~lmapDir  = ~lbaseDir ++ "latent_maps/";

	SynthDef(\clip, { |buf, start = 0, dur = 0.3, amp = 0.5, gate = 1, out = 0,
		lpf = 20000, atk = 0.03, rel = 0.03|
		var live = Line.kr(1, 0, (dur - rel).max(0.01)) > 0;
		var env = EnvGen.kr(Env.asr(atk, 1, rel), gate * live, doneAction: 2);
		var sig = PlayBuf.ar(1, buf, BufRateScale.kr(buf), startPos: start, loop: 0);
		sig = LPF.ar(LPF.ar(sig, lpf), lpf);
		Out.ar(out, Pan2.ar(sig * env, 0, amp));
	}).add;
	s.sync;

	~lamp = 0.5;
	~llenFrac = 1.0;
	~lclipDur = 0.3;
	~llpf = 20000;
	~lfade = 0.03;
	~lhover = true;
	~lloop = false;
	~lrev = false;
	~levenSp = true;
	~lloading = false;
	~lkeyKind = 0;      // 0 brightness, 1 level, 2 PC1, 3.. latent dim (kind - 3)
	~lclSel = 0;        // index into ~lgroupIds
	~lval = 0.5;
	~lcurIdx = 0;
	~lline = [];  ~llx = [];  ~lly = [];
	~llast = nil;  ~lbuf = nil;
	~lloopR = nil;  ~lsweepR = nil;
	~lbyGroup = Dictionary.new;
	~lgroupIds = [];
	~lnDim = 0;
	~lrefreshGui = nil;
	~lkeyName = "brightness";

	~lfindMaps = {
		PathName(~lmapDir).files
			.select { |f| f.fileName.endsWith("_map.csv") }
			.collect { |f| f.fileName.replace("_map.csv", "") }
			.sort
	};
	~lmaps = ~lfindMaps.value;

	~lpalFor = { |n|
		(if(n <= 10) {
			["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
				"#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
		} {
			["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a", "#d62728", "#ff9896",
				"#9467bd", "#c5b0d5", "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7",
				"#bcbd22", "#dbdb8d", "#17becf", "#9edae5"]
		}).collect { |h| Color.fromHexString(h) }
	};
	~lpal = ~lpalFor.(8);

	// ---- small maths helpers
	~lcorr = { |a, b|
		var da, db;
		da = a - a.mean;
		db = b - b.mean;
		(da * db).sum / (((da.squared.sum) * (db.squared.sum)).sqrt + 1e-12)
	};
	~lrank = { |a| a.order.order };   // rank of each element (ties are not averaged)
	~lnorm = { |a|
		var lo = a.minItem, hi = a.maxItem;
		a.collect { |v| if(hi > lo) { (v - lo) / (hi - lo) } { 0.5 } }
	};
	// first principal axis of the latent vectors of the given clips: the position of each clip along it
	~lpc1 = { |ids|
		var zz, m, xx, v, u;
		if(ids.size < 3) {
			ids.collect { 0.0 }
		} {
			zz = ids.collect { |i| ~lrows[i][9..] };
			m = zz.flop.collect { |c| c.mean };
			xx = zz.collect { |r| r - m };
			v = Array.fill(~lnDim) { 1.0.rand2 };
			40.do {
				u = xx.collect { |r| (r * v).sum };
				v = xx.flop.collect { |c| (c * u).sum };
				v = v / (v.squared.sum.sqrt + 1e-12);
			};
			xx.collect { |r| (r * v).sum }
		}
	};

	// ---- load a map
	~lloadMap = { |name|
		{
			var base = ~lmapDir ++ name;
			var buf, meta, gn;
			~lloading = true;
			~lcurMap = name;
			if(~llast.notNil and: { ~llast.isPlaying }) { ~llast.set(\gate, 0) };

			~lrows = CSVFileReader.read(base ++ "_map.csv", true, true)
				.collect { |r| r.collect { |x| x.asFloat } };
			~lgrp  = ~lrows.collect { |r| r[6].asInteger };
			~lcen  = ~lrows.collect { |r| r[7] };
			~llvl  = ~lrows.collect { |r| r[8] };
			~linfo = File.readAllString(base ++ "_map_info.txt").split($\n);
			~lnDim = ~lrows[0].size - 9;
			gn = Dictionary.new;
			meta = Dictionary.new;
			File.readAllString(base ++ "_map_groups.txt").split($\n).do { |l|
				var p = l.split($|);
				if(p.size == 2) {
					if(p[0].beginsWith("#")) { meta[p[0]] = p[1] } { gn[p[0].asInteger] = p[1] };
				};
			};
			~lgnames = gn;
			~lclipLen = meta["#clip_samples"].asInteger;
			~lclipDur = ~lclipLen / meta["#sr"].asFloat;

			buf = Buffer.read(s, ~lmapDir ++ meta["#audio"]);
			s.sync;
			if(~lbuf.notNil) { ~lbuf.free };
			~lbuf = buf;

			~lbyGroup = Dictionary.new;
			~lgrp.do { |g, i| ~lbyGroup[g] = (~lbyGroup[g] ? []).add(i) };
			~lgroupIds = ~lbyGroup.keys.asArray.sort;
			~lpal = ~lpalFor.(~lgroupIds.size);
			~lclSel = 0;
			if(~lkeyKind > (2 + ~lnDim)) { ~lkeyKind = 0 };
			~lloading = false;
			~lrebuild.value;
			{ ~lrefreshGui.value }.defer;
			"cluster line ready: %  (% clips, % clusters)".format(name, ~lrows.size, ~lgroupIds.size).postln;
		}.fork;
	};

	// ---- sort the clips of the chosen cluster along the chosen key
	~lrebuild = {
		var ids, cenv, lvlv, kv, ord, n, other, rk, txt, dirTxt;
		if(~lloading.not and: { ~lgroupIds.size > 0 }) {
			ids  = ~lbyGroup[~lgroupIds[~lclSel]];
			cenv = ids.collect { |i| ~lcen[i] };
			lvlv = ids.collect { |i| ~llvl[i] };
			kv = case
				{ ~lkeyKind == 0 } { cenv }
				{ ~lkeyKind == 1 } { lvlv }
				{ ~lkeyKind == 2 } {
					var p = ~lpc1.(ids);
					if(~lcorr.(p, cenv) < 0) { p.neg } { p }   // orient towards brighter
				}
				{ ids.collect { |i| ~lrows[i][9 + (~lkeyKind - 3)] } };
			if(~lrev) { kv = kv.neg };
			ord = kv.order;
			ids = ids[ord];
			kv = kv[ord];
			cenv = cenv[ord];
			lvlv = lvlv[ord];
			n = ids.size;
			~lline = ids;
			~llx = if(~levenSp) {
				n.collect { |j| if(n > 1) { j / (n - 1) } { 0.5 } }
			} {
				~lnorm.(kv)
			};
			other = if(~lkeyKind == 1) { cenv } { lvlv };
			~lly = ~lnorm.(other);
			~lotherName = if(~lkeyKind == 1) { "brightness" } { "level" };
			~lkeyName = case
				{ ~lkeyKind == 0 } { "brightness" }
				{ ~lkeyKind == 1 } { "level" }
				{ ~lkeyKind == 2 } { "main latent axis" }
				{ "latent dim " ++ (~lkeyKind - 3) };
			dirTxt = if(~lrev) { "highest" } { "lowest" };
			~lcurIdx = (~llx - ~lval).abs.minIndex;
			if(n > 2) {
				rk = ~lrank.(kv);
				txt = "% clips. Along the line (rank correlation): brightness %, level %".format(
					n, ~lcorr.(rk, ~lrank.(cenv)).round(0.01), ~lcorr.(rk, ~lrank.(lvlv)).round(0.01));
			} {
				txt = "% clips".format(n);
			};
			~lcorrText = txt;
			~lendText = "value 0 = % %, value 1 = the other end.  Dot height = %".format(dirTxt, ~lkeyName, ~lotherName);
			{ ~lshowCur.value }.defer;
		};
	};

	~lshowCur = {
		if(~llabel.notNil and: { ~lline.size > 0 }) {
			~llabel.string = "value %  ->  clip % of %   |   %   |   brightness % Hz, level % dB".format(
				~lval.round(0.01), ~lcurIdx + 1, ~lline.size, ~linfo[~lline[~lcurIdx]],
				~lcen[~lline[~lcurIdx]].round(1), ~llvl[~lline[~lcurIdx]].round(1));
			~lcorrLabel.string = ~lcorrText;
			~lendLabel.string = ~lendText;
		};
		if(~luv.notNil) { ~luv.refresh };
	};

	~lplay = { |i|
		var d = ~lclipDur * ~llenFrac;
		var f = min(~lfade, d / 2);
		if(~llast.notNil and: { ~llast.isPlaying }) { ~llast.set(\gate, 0) };
		~llast = Synth(\clip, [\buf, ~lbuf, \start, i * ~lclipLen, \dur, d, \amp, ~lamp,
			\lpf, ~llpf, \atk, f, \rel, f]).register;
	};

	// the value 0..1 (this is the function Sentire's proximity value would call)
	~ltick = { |v, play = true|
		var idx;
		if(~lline.size > 0) {
			~lval = v.clip(0, 1);
			idx = (~llx - ~lval).abs.minIndex;
			if(idx != ~lcurIdx) {
				~lcurIdx = idx;
				if(play and: { ~lloop.not }) { ~lplay.(~lline[idx]) };
			};
			{ ~lshowCur.value }.defer;
		};
	};

	~lsetLoop = { |on|
		~lloop = on;
		if(~lloopR.notNil) { ~lloopR.stop; ~lloopR = nil };
		if(on) {
			~lloopR = Routine({
				loop {
					var d, f;
					if(~lline.size > 0) { ~lplay.(~lline[~lcurIdx]) };
					d = ~lclipDur * ~llenFrac;
					f = min(~lfade, d / 2);
					(d - f).max(0.05).wait;
				};
			}).play(SystemClock);
		};
	};

	~lsweep = { |secs|
		if(~lsweepR.notNil) { ~lsweepR.stop; ~lsweepR = nil };
		if(secs.notNil) {
			~lsweepR = Routine({
				var steps = (secs / 0.02).asInteger.max(2);
				(2 * steps).do { |k|
					var v = if(k < steps) { k / (steps - 1) } { 1 - ((k - steps) / (steps - 1)) };
					~ltick.(v);
					{ ~lvalSl.value = v }.defer;
					0.02.wait;
				};
				{ ~lsweepBtn.value = 0 }.defer;
			}).play(SystemClock);
		};
	};

	~lfromX = { |x, trigger|
		var w = ~luv.bounds.width;
		var v = ((x - 50) / (w - 100)).clip(0, 1);
		{ ~lvalSl.value = v }.defer;
		~ltick.(v, trigger);
	};

	// ---- the window
	~lopenWindow = {
		if(~lwin.notNil and: { ~lwin.isClosed.not }) { ~lwin.close };

		~luv = UserView().minSize_(Size(600, 380)).background_(Color.white).acceptsMouseOver_(true);
		~luv.drawFunc = { |v|
			var w = v.bounds.width, h = v.bounds.height;
			var l = 50, r = w - 50, top = 30, bot = h - 50;
			var px, py;
			px = { |x| l + (x * (r - l)) };
			py = { |y| bot - ((0.05 + (0.9 * y)) * (bot - top)) };
			Pen.fillColor = Color.gray(0.97);
			Pen.fillRect(Rect(l - 10, top - 10, r - l + 20, bot - top + 20));
			if(~lloading.not and: { ~lline.size > 0 }) {
				Pen.strokeColor = Color.gray(0.6);
				Pen.width = 1;
				[0, 0.25, 0.5, 0.75, 1].do { |t|
					Pen.line(Point(px.(t), bot + 4), Point(px.(t), bot + 10));
					Pen.stringAtPoint(t.asString, Point(px.(t) - 8, bot + 12), Font("Helvetica", 11), Color.gray(0.3));
				};
				Pen.line(Point(l, bot + 4), Point(r, bot + 4));
				Pen.stroke;
				// dots in the colour of the cluster (same colours as in rave_sound_map.scd)
				Pen.fillColor = ~lpal[~lgroupIds[~lclSel] % ~lpal.size];
				~lline.size.do { |j|
					Pen.addOval(Rect(px.(~llx[j]) - 3.5, py.(~lly[j]) - 3.5, 7, 7));
				};
				Pen.fill;
				// the value
				Pen.strokeColor = Color.black;
				Pen.width = 1;
				Pen.line(Point(px.(~lval), top - 8), Point(px.(~lval), bot + 4));
				Pen.stroke;
				Pen.width = 2;
				Pen.strokeOval(Rect(px.(~llx[~lcurIdx]) - 8, py.(~lly[~lcurIdx]) - 8, 16, 16));
			};
		};
		~luv.mouseOverAction = { |v, x, y| if(~lhover) { ~lfromX.(x, true) } };
		~luv.mouseMoveAction = { |v, x, y| ~lfromX.(x, true) };
		~luv.mouseDownAction = { |v, x, y| ~lfromX.(x, true) };

		~llabel = StaticText().string_("move the mouse over the line, or use the slider");
		~lcorrLabel = StaticText().string_("");
		~lendLabel = StaticText().string_("");

		~lvalSl = Slider().orientation_(\horizontal).value_(0.5).minSize_(Size(100, 28))
			.action_({ |sl| ~ltick.(sl.value, true) });

		~lmapPop = PopUpMenu().items_(~lmaps).action_({ |m| ~lloadMap.(~lmaps[m.value]) });
		~lrescanBtn = Button().states_([["rescan folder"]]).action_({
			~lmaps = ~lfindMaps.value;
			~lmapPop.items = ~lmaps;
			~lmapPop.value = ~lmaps.indexOfEqual(~lcurMap) ? 0;
		});
		~lclPop = PopUpMenu().items_(["-"]).action_({ |m| ~lclSel = m.value; ~lrebuild.value });
		~lkeyPop = PopUpMenu().items_(["brightness (spectral centroid)", "level (dB)", "main latent axis (PC1)"])
			.action_({ |m| ~lkeyKind = m.value; ~lrebuild.value });
		~lspBtn = Button().states_([["spacing: even"], ["spacing: by value"]])
			.action_({ |b| ~levenSp = (b.value == 0); ~lrebuild.value });
		~lrevBtn = Button().states_([["reverse: off"], ["reverse: on"]])
			.action_({ |b| ~lrev = (b.value == 1); ~lrebuild.value });
		~lhoverBtn = Button().states_([["play on hover: off"], ["play on hover: on"]]).value_(1)
			.action_({ |b| ~lhover = (b.value == 1) });
		~lloopBtn = Button().states_([["loop: off"], ["loop: on"]])
			.action_({ |b| ~lsetLoop.(b.value == 1) });
		~lsweepBox = NumberBox().value_(8).clipLo_(1).clipHi_(60).decimals_(1);
		~lsweepBtn = Button().states_([["sweep up and down"], ["stop sweep"]])
			.action_({ |b| if(b.value == 1) { ~lsweep.(~lsweepBox.value) } { ~lsweep.(nil) } });

		~lampSpec = ControlSpec(0.1, 4, \exp);
		~lampSl = Slider().orientation_(\horizontal).value_(~lampSpec.unmap(0.5))
			.action_({ |sl| ~lamp = ~lampSpec.map(sl.value) });
		~llpfSpec = ControlSpec(500, 20000, \exp);
		~lfadeSpec = ControlSpec(0.002, 0.15, \exp);
		~llpfLabel = StaticText().string_("low-pass: off");
		~llpfSl = Slider().orientation_(\horizontal).value_(1.0).action_({ |sl|
			~llpf = ~llpfSpec.map(sl.value);
			~llpfLabel.string = if(~llpf > 19000) { "low-pass: off" } { "low-pass: % Hz".format(~llpf.round(10).asInteger) };
		});
		~lfadeLabel = StaticText().string_("fade in / out: 30 ms");
		~lfadeSl = Slider().orientation_(\horizontal).value_(~lfadeSpec.unmap(0.03)).action_({ |sl|
			~lfade = ~lfadeSpec.map(sl.value);
			~lfadeLabel.string = "fade in / out: % ms".format((~lfade * 1000).round(1));
		});
		~llenLabel = StaticText().string_("sample length");
		~llenSl = Slider().orientation_(\horizontal).value_(1.0).action_({ |sl|
			~llenFrac = sl.value.linlin(0, 1, 0.1, 1);
			~llenLabel.string = ~llenText.value;
		});

		~lwin = Window("RAVE cluster line", Rect(60, 60, 1050, 640));
		~lwin.layout = HLayout(
			[VLayout(~luv, ~lvalSl, ~llabel, ~lcorrLabel, ~lendLabel), stretch: 3],
			[VLayout(
				StaticText().string_("map (model)"), HLayout(~lmapPop, ~lrescanBtn),
				StaticText().string_("cluster"), ~lclPop,
				StaticText().string_("order by"), ~lkeyPop,
				HLayout(~lspBtn, ~lrevBtn),
				~lhoverBtn, ~lloopBtn,
				HLayout(~lsweepBtn, StaticText().string_("seconds"), ~lsweepBox),
				StaticText().string_("volume"), ~lampSl,
				StaticText().string_("sample length"), ~llenSl, ~llenLabel,
				StaticText().string_("smoothing"), ~llpfSl, ~llpfLabel, ~lfadeSl, ~lfadeLabel,
				nil
			), stretch: 1]
		);
		~lwin.acceptsMouseOver = true;
		~lwin.onClose = { ~lsetLoop.(false); ~lsweep.(nil) };
		~lwin.front;
		~llenText = { "length: % s (of % s stored)".format((~lclipDur * ~llenFrac).round(0.01), ~lclipDur.round(0.01)) };
		~lrefreshGui = {
			if(~lwin.notNil and: { ~lwin.isClosed.not }) {
				~lmapPop.items = ~lmaps;
				~lmapPop.value = ~lmaps.indexOfEqual(~lcurMap) ? 0;
				if(~lgroupIds.size > 0) {
					~lclPop.items = ~lgroupIds.collect { |g| ~lgnames[g] };
					~lclPop.value = ~lclSel;
				};
				~lkeyPop.items = ["brightness (spectral centroid)", "level (dB)", "main latent axis (PC1)"]
					++ ~lnDim.collect { |d| "latent dim " ++ d };
				~lkeyPop.value = ~lkeyKind;
				~llenLabel.string = ~llenText.value;
				~lshowCur.value;
			};
		};
		~lrefreshGui.value;
	};

	if(~lmaps.isEmpty) {
		"No maps found in %. Make one with latent_map.py or with rave_sound_map.scd.".format(~lmapDir).postln;
	} {
		~lloadMap.(~lmaps.detect { |m| m.contains("dawnchorus") } ? ~lmaps[0]);
	};
	{ ~lopenWindow.value }.defer;
};
)

// 2. Reopen the window (if you closed it)
~lopenWindow.value;

// Use the value from your own code (0..1), for example from a Sentire proximity value:
// ~ltick.(0.3);

// Stop everything
// ~lsetLoop.(false); ~lsweep.(nil); ~llast.free; ~lbuf.free;
