--
-- PostgreSQL database dump
--

-- Dumped from database version 17.2
-- Dumped by pg_dump version 17.2

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: alembic_version; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.alembic_version (
    version_num character varying(32) NOT NULL
);


ALTER TABLE public.alembic_version OWNER TO postgres;

--
-- Name: attendance_session; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.attendance_session (
    id integer NOT NULL,
    session_number integer NOT NULL,
    teacher_id character varying(11) NOT NULL,
    course_id integer NOT NULL,
    ip_address character varying(45) NOT NULL,
    start_time timestamp without time zone NOT NULL,
    end_time timestamp without time zone,
    is_active boolean,
    status character varying(20),
    wifi_ssid character varying NOT NULL
);


ALTER TABLE public.attendance_session OWNER TO postgres;

--
-- Name: attendance_session_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.attendance_session_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.attendance_session_id_seq OWNER TO postgres;

--
-- Name: attendance_session_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.attendance_session_id_seq OWNED BY public.attendance_session.id;


--
-- Name: attendancelog; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.attendancelog (
    student_id character varying(11) NOT NULL,
    course_id integer NOT NULL,
    session_id integer NOT NULL,
    teacher_id character varying(11) NOT NULL,
    connection_strength character varying(20) NOT NULL,
    date date NOT NULL,
    "time" time without time zone NOT NULL,
    status character varying(10) NOT NULL,
    marking_ip character varying(45),
    verification_score double precision,
    liveness_score double precision,
    verification_method character varying(20),
    verification_timestamp timestamp without time zone,
    attempts_count integer,
    last_attempt timestamp without time zone,
    verification_details json
);


ALTER TABLE public.attendancelog OWNER TO postgres;

--
-- Name: course; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.course (
    course_id integer NOT NULL,
    course_name character varying(255) NOT NULL,
    sessions integer,
    teacher_id character varying(11)
);


ALTER TABLE public.course OWNER TO postgres;

--
-- Name: course_course_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.course_course_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.course_course_id_seq OWNER TO postgres;

--
-- Name: course_course_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.course_course_id_seq OWNED BY public.course.course_id;


--
-- Name: student; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.student (
    student_id character varying(11) NOT NULL,
    name character varying(255) NOT NULL,
    password character varying(255) NOT NULL,
    face_encoding double precision[],
    email character varying(255) NOT NULL
);


ALTER TABLE public.student OWNER TO postgres;

--
-- Name: student_courses; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.student_courses (
    student_id character varying(11) NOT NULL,
    course_id integer NOT NULL
);


ALTER TABLE public.student_courses OWNER TO postgres;

--
-- Name: teacher; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.teacher (
    teacher_id character varying(11) NOT NULL,
    name character varying(255) NOT NULL,
    password character varying(255) NOT NULL
);


ALTER TABLE public.teacher OWNER TO postgres;

--
-- Name: attendance_session id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.attendance_session ALTER COLUMN id SET DEFAULT nextval('public.attendance_session_id_seq'::regclass);


--
-- Name: course course_id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.course ALTER COLUMN course_id SET DEFAULT nextval('public.course_course_id_seq'::regclass);


--
-- Data for Name: alembic_version; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.alembic_version (version_num) FROM stdin;
837ac3695f20
\.


--
-- Data for Name: attendance_session; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.attendance_session (id, session_number, teacher_id, course_id, ip_address, start_time, end_time, is_active, status, wifi_ssid) FROM stdin;
1	1	123	2	127.0.0.1	2025-06-13 06:33:13.540272	2025-06-13 06:33:39.834906	f	completed	UNKNOWN
3	3	123	2	127.0.0.1	2025-06-13 08:02:58.525725	2025-06-13 09:38:57.079356	f	completed	Classroom_WiFi123
2	2	123	2	127.0.0.1	2025-06-13 06:49:06.776496	2025-06-13 09:39:00.355658	f	completed	UNKNOWN
4	4	123	2	127.0.0.1	2025-06-13 08:06:50.585555	2025-06-13 09:39:07.033907	f	completed	Classroom_WiFi123
\.


--
-- Data for Name: attendancelog; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.attendancelog (student_id, course_id, session_id, teacher_id, connection_strength, date, "time", status, marking_ip, verification_score, liveness_score, verification_method, verification_timestamp, attempts_count, last_attempt, verification_details) FROM stdin;
20221442260	2	4	123	none	2025-06-13	08:11:46.750547	present	\N	0.6567420910778698	0.9991565346717834	face	2025-06-13 08:11:46.750547	1	2025-06-13 08:11:46.750547	{}
\.


--
-- Data for Name: course; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.course (course_id, course_name, sessions, teacher_id) FROM stdin;
2	Introduction to AI	0	123
\.


--
-- Data for Name: student; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.student (student_id, name, password, face_encoding, email) FROM stdin;
20221442260	Khaled	$2b$12$kSQACvNT/Ss6JSbxhjPfkupPQqhjP2JSIu2Pc/ARrAyRLLUMmyPpy	{-0.0661350386683953,0.018292299976108362,-0.0005102012286542943,0.1762878846594193,0.016218175807653664,0.14103672129514574,0.08123595934807362,0.03159902610411598,0.08575834398087041,0.053557564599525105,0.04409686619368698,-0.20633035439259761,0.003435934202777281,-0.13921260630529306,0.020710422794005822,-0.029724136214589882,-0.04818824545818979,0.0013938772553051453,-0.08338671607256162,-0.1298266917169799,-0.12860551724497407,0.05060468738032732,-0.011675914150547393,0.049784074563982546,-0.014603041083378784,-0.006482588621104556,0.1051154506943479,-0.0034690418019724914,-0.057192569557965055,-0.10354420561464757,-0.1426385927398817,-0.021138263832836904,-0.06505883590206346,-0.029342972737750158,0.22618915295421194,-0.03859326544878676,-0.11577093770482957,0.03304788666918368,0.0890680252639053,0.01464369358435693,0.07875697625594312,-0.11763300490203137,0.07105697790036261,-0.0014977490850281616,0.05864641850824689,-0.1587537081634527,0.03737350776703968,0.023634297773586534,0.04030879852178881,-0.09188422912037315,-0.020156044366656516,0.0834306217609283,0.053058874174654144,0.15835317612710018,-0.06550494218116402,0.09789863705561498,0.08911266156188276,-0.20382392901110008,-0.2094794002922896,0.003743860853894609,-0.01991785946285326,-0.07144034061086181,0.061526570451863605,-0.025627245290421968,0.06636789579907401,0.006328645350286052,0.024916991549440517,0.06919076893651593,-0.12170321129747586,-0.019683580605364386,-0.023531491621902212,0.02688059867288632,-0.05313924123183984,-0.004764772393068817,0.07568438272815747,-0.027627024994428474,-0.06579836586126603,0.11433699786644148,-0.03481260260402486,0.08423695807055434,0.17069315270695296,-0.026481150209571428,-0.04714780800692616,-0.047866935199092575,0.06844749707133384,-0.09808928667161693,0.03612156377000693,-0.09098070198630781,-0.05342295471529048,-0.08876810014557764,-0.048016512707245315,0.01565870532751031,0.048417069426354944,0.05156621405707539,-0.04601857186864432,0.04419473826222283,-0.16498160220158473,0.05408632849567218,-0.057058152199235976,-0.019175335485211924,-0.005666404508456205,-0.06993605985123053,-0.08297681446160603,0.1013727351203837,0.08573120282114026,0.08342436221372244,-0.16206253048653976,-0.14183338196398038,-0.06271024687925134,-0.005504510921569055,0.0686397214472401,0.060344350307146015,-0.02144142486062311,0.030988073563307683,-0.2090751362230604,-0.03295539544170019,-0.13243605406877448,-0.05828155799248553,-0.014302527281293794,0.14033627401356844,-0.06344731349929324,-0.15038266956312873,0.07957751502401653,0.12228237738366385,-0.007719869209200699,0.2110748937125855,0.09590095291768805,0.09405184910452616}	khaledshbrawy@gmail.com
\.


--
-- Data for Name: student_courses; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.student_courses (student_id, course_id) FROM stdin;
20221442260	2
\.


--
-- Data for Name: teacher; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.teacher (teacher_id, name, password) FROM stdin;
123	Sample Student	$2b$12$kSQACvNT/Ss6JSbxhjPfkupPQqhjP2JSIu2Pc/ARrAyRLLUMmyPpy
\.


--
-- Name: attendance_session_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.attendance_session_id_seq', 4, true);


--
-- Name: course_course_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.course_course_id_seq', 2, true);


--
-- Name: alembic_version alembic_version_pkc; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.alembic_version
    ADD CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num);


--
-- Name: attendance_session attendance_session_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.attendance_session
    ADD CONSTRAINT attendance_session_pkey PRIMARY KEY (id);


--
-- Name: attendancelog attendancelog_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.attendancelog
    ADD CONSTRAINT attendancelog_pkey PRIMARY KEY (student_id, course_id, session_id);


--
-- Name: course course_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.course
    ADD CONSTRAINT course_pkey PRIMARY KEY (course_id);


--
-- Name: student_courses student_courses_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.student_courses
    ADD CONSTRAINT student_courses_pkey PRIMARY KEY (student_id, course_id);


--
-- Name: student student_email_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.student
    ADD CONSTRAINT student_email_key UNIQUE (email);


--
-- Name: student student_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.student
    ADD CONSTRAINT student_pkey PRIMARY KEY (student_id);


--
-- Name: student student_student_id_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.student
    ADD CONSTRAINT student_student_id_key UNIQUE (student_id);


--
-- Name: teacher teacher_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.teacher
    ADD CONSTRAINT teacher_pkey PRIMARY KEY (teacher_id);


--
-- Name: teacher teacher_teacher_id_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.teacher
    ADD CONSTRAINT teacher_teacher_id_key UNIQUE (teacher_id);


--
-- Name: attendance_session attendance_session_course_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.attendance_session
    ADD CONSTRAINT attendance_session_course_id_fkey FOREIGN KEY (course_id) REFERENCES public.course(course_id);


--
-- Name: attendance_session attendance_session_teacher_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.attendance_session
    ADD CONSTRAINT attendance_session_teacher_id_fkey FOREIGN KEY (teacher_id) REFERENCES public.teacher(teacher_id);


--
-- Name: attendancelog attendancelog_course_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.attendancelog
    ADD CONSTRAINT attendancelog_course_id_fkey FOREIGN KEY (course_id) REFERENCES public.course(course_id);


--
-- Name: attendancelog attendancelog_session_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.attendancelog
    ADD CONSTRAINT attendancelog_session_id_fkey FOREIGN KEY (session_id) REFERENCES public.attendance_session(id);


--
-- Name: attendancelog attendancelog_student_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.attendancelog
    ADD CONSTRAINT attendancelog_student_id_fkey FOREIGN KEY (student_id) REFERENCES public.student(student_id);


--
-- Name: attendancelog attendancelog_teacher_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.attendancelog
    ADD CONSTRAINT attendancelog_teacher_id_fkey FOREIGN KEY (teacher_id) REFERENCES public.teacher(teacher_id);


--
-- Name: course course_teacher_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.course
    ADD CONSTRAINT course_teacher_id_fkey FOREIGN KEY (teacher_id) REFERENCES public.teacher(teacher_id);


--
-- Name: student_courses student_courses_course_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.student_courses
    ADD CONSTRAINT student_courses_course_id_fkey FOREIGN KEY (course_id) REFERENCES public.course(course_id);


--
-- Name: student_courses student_courses_student_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.student_courses
    ADD CONSTRAINT student_courses_student_id_fkey FOREIGN KEY (student_id) REFERENCES public.student(student_id);


--
-- PostgreSQL database dump complete
--

